"""Runs RAG experiments: every (chunk strategy x embedding model) combination, in parallel."""
import asyncio
import logging
import os
import threading
import time
import uuid
from functools import lru_cache
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .evaluator import evaluate_answer
from .utils import calculate_cost_estimate, mean_or_none, retrieval_hit

log = logging.getLogger("rag-lab")

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer is not in the context, say that you don't know.

Context:
{context}

Question: {question}"""
)

# --- Load heavy objects ONCE (the original reloaded the embedding model for every strategy) ---
_embedders: dict = {}
_embedders_lock = threading.Lock()


def get_embedder(name: str) -> HuggingFaceEmbeddings:
    with _embedders_lock:
        if name not in _embedders:
            log.info("Loading embedding model %s", name)
            _embedders[name] = HuggingFaceEmbeddings(
                model_name=name, encode_kwargs={"normalize_embeddings": True}
            )
        return _embedders[name]


@lru_cache(maxsize=8)
def get_generator(model_name: Optional[str] = None) -> ChatGroq:
    model = model_name or config.GENERATOR_MODEL
    return ChatGroq(model=model, temperature=0, max_retries=3)


# One shared in-memory Chroma client for the whole process. Creating several clients at the same
# time from different threads causes "Could not connect to tenant default_tenant".
_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            _chroma_client = chromadb.EphemeralClient()
        return _chroma_client


def load_documents(file_path: str) -> list[Document]:
    """Loads text from supported file formats (PDF, TXT, MD)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs = [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: PDF, TXT, MD.")

    if not docs or not any(d.page_content.strip() for d in docs):
        raise ValueError("No text found in this file (scanned PDFs need OCR first).")
    return docs


def run_one_experiment(
    docs: list[Document],
    strategy: dict,
    embedder_name: str,
    questions: list[dict],
    top_k: Optional[int] = None,
    generator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> dict:
    """Blocking function (runs in a worker thread). One chunking strategy + one embedder."""
    started = time.perf_counter()
    k_val = top_k or config.TOP_K

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=strategy["chunk"], chunk_overlap=strategy["overlap"]
    )
    splits = splitter.split_documents(docs)
    if not splits:
        raise ValueError(
            f"Strategy '{strategy['name']}' produced 0 chunks. Ensure chunk size is appropriate."
        )

    # Unique collection per experiment -> experiments can never mix their chunks.
    store = Chroma.from_documents(
        documents=splits,
        embedding=get_embedder(embedder_name),
        collection_name=f"exp_{uuid.uuid4().hex}",
        client=get_chroma_client(),
    )
    index_seconds = round(time.perf_counter() - started, 2)

    details = []
    try:
        retriever = store.as_retriever(search_kwargs={"k": k_val})
        llm = get_generator(generator_model)

        for item in questions:
            t0 = time.perf_counter()
            retrieved = retriever.invoke(item["question"])
            context = "\n\n".join(d.page_content for d in retrieved)
            message = llm.invoke(
                ANSWER_PROMPT.format_messages(context=context, question=item["question"])
            )
            latency = round(time.perf_counter() - t0, 2)
            usage = getattr(message, "usage_metadata", None) or {}

            answer_content = message.content if hasattr(message, "content") else str(message)
            verdict = evaluate_answer(
                item["question"], answer_content, context, judge_model=judge_model
            )
            details.append(
                {
                    "question": item["question"],
                    "expected": item["expected"],
                    "answer": answer_content,
                    "context": context,
                    "retrieved_chunks": [d.page_content for d in retrieved],
                    "latency_s": latency,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "retrieval_hit": retrieval_hit(context, item["expected"]),
                    **verdict,
                }
            )
    finally:
        try:
            store.delete_collection()  # free the memory
        except Exception:  # noqa: BLE001
            pass

    avg_rel = mean_or_none(d["relevance_score"] for d in details)
    avg_faith = mean_or_none(d["faithfulness_score"] for d in details)
    hits = [1.0 if d["retrieval_hit"] else 0.0 for d in details if d["retrieval_hit"] is not None]

    tot_in = sum(d["input_tokens"] for d in details)
    tot_out = sum(d["output_tokens"] for d in details)
    cost_info = calculate_cost_estimate(tot_in, tot_out)

    summary = {
        "strategy": strategy["name"],
        "embedder": embedder_name,
        "chunk_size": strategy["chunk"],
        "overlap": strategy["overlap"],
        "top_k": k_val,
        "num_chunks": len(splits),
        "avg_relevance": avg_rel,
        "avg_faithfulness": avg_faith,
        "overall": mean_or_none([avg_rel, avg_faith]),
        "retrieval_hit_rate": mean_or_none(hits),
        "avg_latency_s": mean_or_none(d["latency_s"] for d in details),
        "index_seconds": index_seconds,
        "total_input_tokens": tot_in,
        "total_output_tokens": tot_out,
        "total_tokens": cost_info["total_tokens"],
        "est_commercial_usd": cost_info["estimated_commercial_usd"],
    }
    return {"summary": summary, "details": details}


async def run_experiments(
    file_path: str,
    questions: list[dict],
    strategies: list[dict],
    embedders: list[str],
    top_k: Optional[int] = None,
    generator_model: Optional[str] = None,
    judge_model: Optional[str] = None,
):
    """Load the document once, then run all experiments in parallel (limited by MAX_PARALLEL)."""
    docs = await asyncio.to_thread(load_documents, file_path)

    semaphore = asyncio.Semaphore(config.MAX_PARALLEL)  # protects the Groq rate limit

    async def guarded(strat, emb):
        async with semaphore:
            return await asyncio.to_thread(
                run_one_experiment,
                docs,
                strat,
                emb,
                questions,
                top_k=top_k,
                generator_model=generator_model,
                judge_model=judge_model,
            )

    experiment_plan = [(s, e) for e in embedders for s in strategies]
    tasks = [guarded(s, e) for s, e in experiment_plan]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    errors = []
    for (strat, emb), outcome in zip(experiment_plan, raw):
        if isinstance(outcome, Exception):
            errors.append(f"[{strat['name']} | {emb}]: {type(outcome).__name__}: {outcome}")
        else:
            results.append(outcome)

    return results, errors
