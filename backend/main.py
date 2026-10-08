"""FastAPI entry point.  Run from the project root:  uvicorn backend.main:app --reload --port 8000"""
import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .pipelines import run_experiments
from .utils import (
    export_markdown_report,
    generate_sample_dataset,
    parse_embedders,
    parse_questions,
    parse_strategies,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = FastAPI(
    title="RAG Evaluation Tool",
    description="Automated comparative benchmarking for RAG pipelines (chunking, embeddings, retrieval hit-rate, faithfulness, relevance).",
    version="2.0.0",
)

# Wildcard origins cannot be combined with credentials, so credentials are off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static directory if present
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if "application/json" in accept and "text/html" not in accept:
        return {
            "status": "RAG Evaluation Tool backend is running",
            "version": "2.0.0",
            "docs": "/docs",
        }
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "RAG Evaluation Tool backend is running",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/api/info")
def api_info():
    return {
        "status": "RAG Evaluation Tool backend is running",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "api_key_set": bool(config.GROQ_API_KEY),
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
    }


@app.get("/config")
def get_config():
    """The frontend reads defaults, models, and presets from here so they live in ONE place."""
    return {
        "allowed_embedders": config.ALLOWED_EMBEDDERS,
        "default_strategies": config.DEFAULT_STRATEGIES,
        "strategy_presets": config.STRATEGY_PRESETS,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "supported_generator_models": config.SUPPORTED_GENERATOR_MODELS,
        "supported_judge_models": config.SUPPORTED_JUDGE_MODELS,
        "top_k": config.TOP_K,
        "max_questions": config.MAX_QUESTIONS,
        "max_experiments": config.MAX_EXPERIMENTS,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "allowed_extensions": list(config.ALLOWED_EXTENSIONS),
    }


@app.get("/sample")
def get_sample_benchmark():
    """Returns pre-built sample document and benchmark questions for instant 1-click testing."""
    return generate_sample_dataset()


@app.post("/optimize")
async def optimize(
    file: UploadFile = File(...),
    questions: str = Form(""),
    strategies: str = Form(""),
    embedders: str = Form(""),
    top_k: Optional[int] = Form(None),
    generator_model: Optional[str] = Form(None),
    judge_model: Optional[str] = Form(None),
):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise HTTPException(
            400, f"Unsupported file type '{ext}'. Please upload one of: {allowed}"
        )

    # Validate inputs
    try:
        question_items = parse_questions(questions)
        strategy_list = parse_strategies(strategies)
        embedder_list = parse_embedders(embedders)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid input: {exc}")

    if not question_items:
        raise HTTPException(400, "Provide at least one question.")
    if len(question_items) > config.MAX_QUESTIONS:
        raise HTTPException(400, f"At most {config.MAX_QUESTIONS} questions per run.")
    if len(strategy_list) * len(embedder_list) > config.MAX_EXPERIMENTS:
        raise HTTPException(
            400,
            f"At most {config.MAX_EXPERIMENTS} experiments (strategies x embedders) per run.",
        )

    retrieval_k = top_k or config.TOP_K
    if not 1 <= retrieval_k <= 20:
        raise HTTPException(400, "top_k must be between 1 and 20.")

    # Checked AFTER input validation so bad requests always get a 400 (even without an API key, e.g. in CI)
    if not config.GROQ_API_KEY:
        raise HTTPException(
            500, "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    gen_model = generator_model.strip() if generator_model and generator_model.strip() else None
    jdg_model = judge_model.strip() if judge_model and judge_model.strip() else None

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")

    # Unique temp file with proper suffix so loaders know how to parse it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        results, errors = await run_experiments(
            tmp_path,
            question_items,
            strategy_list,
            embedder_list,
            top_k=retrieval_k,
            generator_model=gen_model,
            judge_model=jdg_model,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    settings_used = {
        "generator_model": gen_model or config.GENERATOR_MODEL,
        "judge_model": jdg_model or config.JUDGE_MODEL,
        "top_k": retrieval_k,
        "questions": len(question_items),
        "source_file": filename,
    }

    markdown_report = export_markdown_report(results, settings_used)

    return {
        "results": results,
        "errors": errors,
        "settings": settings_used,
        "markdown_report": markdown_report,
    }
