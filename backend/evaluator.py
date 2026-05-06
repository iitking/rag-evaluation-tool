"""LLM-as-a-judge: scores each answer for relevance and faithfulness."""
import json
import logging
import re
from functools import lru_cache
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from . import config

log = logging.getLogger("rag-lab")


class Verdict(BaseModel):
    relevance_score: int = Field(ge=1, le=10, description="1-10: does the answer directly address the question?")
    faithfulness_score: int = Field(ge=1, le=10, description="1-10: is every claim supported by the context? (10 = no hallucination)")
    explanation: str = Field(description="One or two sentences justifying the scores")


_parser = JsonOutputParser(pydantic_object=Verdict)

_prompt = ChatPromptTemplate.from_template(
    """You are a strict, impartial judge of a Retrieval-Augmented Generation (RAG) system.

Question: {question}

System answer: {answer}

Retrieved context:
{context}

Score the answer from 1 to 10 on:
1. relevance_score - does the answer directly address the question?
2. faithfulness_score - is every claim in the answer supported by the retrieved context?
   Penalise anything invented. If the answer says it does not know and the context truly lacks
   the information, that is faithful.

{format_instructions}"""
).partial(format_instructions=_parser.get_format_instructions())


@lru_cache(maxsize=8)
def get_judge(model_name: Optional[str] = None) -> ChatGroq:
    model = model_name or config.JUDGE_MODEL
    return ChatGroq(model=model, temperature=0, max_retries=3)


def parse_judge_output(text: str) -> dict:
    """Robust multi-layer JSON parser with regex fallback to guarantee scores are extracted."""
    # 1. Strip reasoning traces if present (e.g. <think>...</think> from deepseek-r1)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # 2. Extract content from markdown code block if present
    code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
    if code_match:
        cleaned = code_match.group(1).strip()
    else:
        # Or find the outer-most JSON curly braces
        curly_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if curly_match:
            cleaned = curly_match.group(1).strip()

    # 3. Try standard json.loads
    try:
        data = json.loads(cleaned)
        rel = int(data.get("relevance_score", 5))
        faith = int(data.get("faithfulness_score", 5))
        return {
            "relevance_score": max(1, min(10, rel)),
            "faithfulness_score": max(1, min(10, faith)),
            "explanation": str(data.get("explanation", "")).strip(),
        }
    except Exception:
        pass

    # 4. Regex fallback if JSON was malformed
    m_rel = re.search(r"[\"']?relevance_score[\"']?\s*[:=]\s*(\d+)", text)
    m_faith = re.search(r"[\"']?faithfulness_score[\"']?\s*[:=]\s*(\d+)", text)
    m_exp = re.search(r"[\"']?explanation[\"']?\s*[:=]\s*[\"']?([^\"'\n\r}]+)", text)

    if m_rel or m_faith:
        rel = int(m_rel.group(1)) if m_rel else 5
        faith = int(m_faith.group(1)) if m_faith else 5
        exp = m_exp.group(1).strip() if m_exp else ""
        return {
            "relevance_score": max(1, min(10, rel)),
            "faithfulness_score": max(1, min(10, faith)),
            "explanation": exp,
        }

    raise ValueError(f"Could not extract scores from judge output: {text[:150]}")


def evaluate_answer(question: str, answer: str, context: str, judge_model: Optional[str] = None) -> dict:
    """Return scores. On failure scores are None (NOT 0) so they don't drag averages down."""
    try:
        judge_llm = get_judge(judge_model)
        formatted_messages = _prompt.format_messages(
            question=question,
            answer=answer,
            context=context[: config.JUDGE_CONTEXT_CHARS],
        )
        response = judge_llm.invoke(formatted_messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse with resilient parser
        parsed = parse_judge_output(content)
        return {
            "relevance_score": parsed["relevance_score"],
            "faithfulness_score": parsed["faithfulness_score"],
            "explanation": parsed["explanation"],
            "judge_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - judge output can be messy; never crash the whole run
        log.warning("Judge evaluation failed for question '%s': %s", question[:50], exc)
        return {
            "relevance_score": None,
            "faithfulness_score": None,
            "explanation": "",
            "judge_error": str(exc)[:200],
        }
