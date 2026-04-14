"""Small, pure helper functions (no LLM / network). Easy to unit-test."""
import json
from typing import Optional

from . import config


def parse_questions(raw: str) -> list[dict]:
    """One question per line. Optional expected phrase after '||'.

    Example line:  What is the notice period? || 30 days
    The phrase is used to check whether retrieval found the right text.
    Multiple alternative phrases can be separated by ';' (e.g. '30 days; one month').
    """
    items = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        question, _, expected = line.partition("||")
        question, expected = question.strip(), expected.strip()
        if question:
            items.append({"question": question, "expected": expected or None})
    return items


def parse_strategies(raw: Optional[str]) -> list[dict]:
    """Parse a JSON list like [{"name": "A", "chunk": 512, "overlap": 50}]. Empty -> defaults."""
    if not raw or not raw.strip():
        return [dict(s) for s in config.DEFAULT_STRATEGIES]

    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("strategies must be a non-empty list")
    if len(data) > config.MAX_STRATEGIES:
        raise ValueError(f"At most {config.MAX_STRATEGIES} strategies are allowed")

    strategies = []
    for item in data:
        chunk = int(item["chunk"])
        overlap = int(item.get("overlap", 0))
        if not 100 <= chunk <= 4000:
            raise ValueError("chunk size must be between 100 and 4000")
        if not 0 <= overlap < chunk:
            raise ValueError("overlap must be >= 0 and smaller than chunk size")
        name = str(item.get("name") or f"chunk{chunk}")
        strategies.append({"name": name, "chunk": chunk, "overlap": overlap})
    return strategies


def parse_embedders(raw: Optional[str]) -> list[str]:
    """Comma separated embedder names. Empty -> first allowed one."""
    if not raw or not raw.strip():
        return [config.ALLOWED_EMBEDDERS[0]]
    names = [n.strip() for n in raw.split(",") if n.strip()]
    bad = [n for n in names if n not in config.ALLOWED_EMBEDDERS]
    if bad:
        raise ValueError(f"Unknown embedding model(s): {bad}. Allowed: {config.ALLOWED_EMBEDDERS}")
    return list(dict.fromkeys(names))  # remove duplicates, keep order


def retrieval_hit(context: str, expected: Optional[str]) -> Optional[bool]:
    """Did the retrieved text contain the expected phrase?

    Returns None if no phrase was given.
    If expected contains semicolons (e.g. '30 days; one month'), returns True
    if any of the alternative phrases is found.
    """
    if not expected:
        return None
    lower_context = context.lower()
    alternatives = [p.strip().lower() for p in expected.split(";") if p.strip()]
    if not alternatives:
        return None
    return any(alt in lower_context for alt in alternatives)


def mean_or_none(values) -> Optional[float]:
    """Average of the non-None values, rounded. None if there is nothing to average."""
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def calculate_cost_estimate(input_tokens: int, output_tokens: int) -> dict:
    """Calculates token counts and commercial cost comparisons (e.g. GPT-4o-mini equivalent)."""
    # Typical baseline rate: $0.15 / 1M input, $0.60 / 1M output
    est_openai_cost = (input_tokens * 0.00000015) + (output_tokens * 0.00000060)
    return {
        "groq_cost_usd": 0.0,  # Free tier on Groq
        "estimated_commercial_usd": round(est_openai_cost, 6),
        "total_tokens": input_tokens + output_tokens,
    }


def generate_sample_dataset() -> dict:
    """Pre-built sample document and benchmark questions for instant 1-click testing."""
    content = """# Comprehensive Guide to Production RAG Architecture

## 1. Fundamentals of Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation enhances Large Language Models by grounding them in authoritative, external knowledge bases before text generation begins. This approach significantly minimizes factual hallucinations and allows enterprise applications to reference proprietary documents without full model fine-tuning.

## 2. Chunking Strategies and Overlap Guidelines
The chunking strategy is the primary determinant of retrieval precision. When dividing documents:
- Small chunks (128 to 256 tokens) provide high semantic specificity and fine-grained retrieval, making them ideal for pinpoint facts and QA.
- Medium chunks (512 tokens) offer a balanced representation of paragraph context and surrounding detail.
- Large chunks (1024+ tokens) capture broad narratives and document sections but risk introducing irrelevant noise into the prompt.
A recommended chunk overlap range of 10% to 20% is essential for preserving sentence context and preventing key facts from being severed at chunk boundaries.

## 3. Dense Retrieval and Embedding Models
Dense retrieval utilizes neural semantic embeddings to map both queries and text chunks into continuous vector spaces. Unlike sparse keyword search like BM25, semantic embeddings understand synonyms, conceptual relationships, and multilingual intent. It is critical that embedding models match the target language and domain to achieve high retrieval hit-rate.

## 4. Evaluation and the Role of LLM Judges
Evaluating RAG systems requires measuring both retrieval and generation quality independently:
- Retrieval Hit-Rate: Verifies whether the necessary source information was retrieved in the top-k chunks.
- Faithfulness: Assesses whether every claim in the generated answer is strictly grounded in the retrieved context, penalizing hallucinations.
- Relevance: Measures whether the response directly addresses the user's inquiry.
An LLM judge evaluates both faithfulness and relevance on a structured 1-10 numerical scale, providing automated, reproducible quality benchmarking.
"""
    questions = (
        "What is the recommended chunk overlap range for preserving sentence context? || 10% to 20%\n"
        "How does dense retrieval differ from sparse keyword search like BM25? || semantic embeddings\n"
        "What are the primary metrics an LLM judge evaluates in RAG? || faithfulness and relevance"
    )
    return {
        "filename": "rag_architecture_guide.md",
        "content": content,
        "questions": questions,
        "strategies": config.DEFAULT_STRATEGIES,
        "embedders": [config.ALLOWED_EMBEDDERS[0], config.ALLOWED_EMBEDDERS[1]],
    }


def export_markdown_report(results: list[dict], settings: dict) -> str:
    """Formats full benchmark results into an executive-ready Markdown report."""
    if not results:
        return "# RAG Evaluation Report\n\nNo successful experiment results."

    # Identify best overall
    scored = [r for r in results if r["summary"].get("overall") is not None]
    best = max(scored, key=lambda r: r["summary"]["overall"]) if scored else results[0]
    bs = best["summary"]

    lines = [
        "# 🧪 RAG Evaluation & Optimization Benchmark Report",
        "",
        "## Executive Summary",
        f"- **Best Performing Strategy:** `{bs['strategy']}` with `{bs['embedder']}`",
        f"- **Overall Quality Score:** `{bs['overall']}/10` (Relevance: `{bs['avg_relevance']}`, Faithfulness: `{bs['avg_faithfulness']}`)",
        f"- **Average Latency:** `{bs['avg_latency_s']}s`",
        f"- **Retrieval Hit Rate:** `{bs['retrieval_hit_rate'] * 100 if bs['retrieval_hit_rate'] is not None else 'N/A'}%`",
        "",
        "## Evaluation Settings",
        f"- **Generator Model:** `{settings.get('generator_model', 'N/A')}`",
        f"- **Judge Model:** `{settings.get('judge_model', 'N/A')}`",
        f"- **Top-K Retrieval:** `{settings.get('top_k', 'N/A')}`",
        f"- **Evaluated Questions:** `{settings.get('questions', len(best['details']))}`",
        "",
        "## Leaderboard Comparison",
        "| Rank | Strategy | Embedder | Overall | Relevance | Faithfulness | Hit Rate | Latency | Chunks | Tokens |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    sorted_res = sorted(
        results,
        key=lambda r: (r["summary"].get("overall") or 0.0, -(r["summary"].get("avg_latency_s") or 999.0)),
        reverse=True,
    )

    for i, r in enumerate(sorted_res, 1):
        s = r["summary"]
        hit_str = f"{round(s['retrieval_hit_rate'] * 100, 1)}%" if s["retrieval_hit_rate"] is not None else "N/A"
        tot_tok = s["total_input_tokens"] + s["total_output_tokens"]
        lines.append(
            f"| #{i} | {s['strategy']} | {s['embedder']} | **{s['overall']}** | "
            f"{s['avg_relevance']} | {s['avg_faithfulness']} | {hit_str} | "
            f"{s['avg_latency_s']}s | {s['num_chunks']} | {tot_tok} |"
        )

    lines.extend([
        "",
        "## Detailed Question-by-Question Analysis",
    ])

    for r in results:
        s = r["summary"]
        lines.extend([
            f"### Experiment: {s['strategy']} ({s['embedder']})",
            f"*Chunk Size: {s['chunk_size']} | Overlap: {s['overlap']} | Overall: {s['overall']}/10*",
            "",
        ])
        for idx, d in enumerate(r["details"], 1):
            hit_label = "✅ Found" if d["retrieval_hit"] is True else ("❌ Missed" if d["retrieval_hit"] is False else "N/A")
            lines.extend([
                f"**Q{idx}: {d['question']}**",
                f"- **Answer:** {d['answer']}",
                f"- **Scores:** Relevance `{d['relevance_score']}/10` | Faithfulness `{d['faithfulness_score']}/10` | Hit `{hit_label}`",
                f"- **Judge Explanation:** {d.get('explanation') or 'N/A'}",
                f"- **Speed & Tokens:** `{d['latency_s']}s` | in: `{d['input_tokens']}`, out: `{d['output_tokens']}`",
                "",
            ])

    return "\n".join(lines)
