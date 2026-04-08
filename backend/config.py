"""Central place for all settings. Everything comes from environment variables / the .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- API key (the ONLY required secret) -------------------------------------
# Get a free key at https://console.groq.com  ->  put it in .env as GROQ_API_KEY=...
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Models (served by Groq) ------------------------------------------------
# Default generator and judge models
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

# Supported models available in UI selection
SUPPORTED_GENERATOR_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b",
]

SUPPORTED_JUDGE_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

# Ensure configured models are always in the list
if GENERATOR_MODEL not in SUPPORTED_GENERATOR_MODELS:
    SUPPORTED_GENERATOR_MODELS.insert(0, GENERATOR_MODEL)
if JUDGE_MODEL not in SUPPORTED_JUDGE_MODELS:
    SUPPORTED_JUDGE_MODELS.insert(0, JUDGE_MODEL)

# --- Retrieval / safety limits -------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))                       # chunks retrieved per question
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "3"))         # experiments running at once
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_QUESTIONS = 10
MAX_EXPERIMENTS = 12                                       # strategies x embedders
MAX_STRATEGIES = 6
JUDGE_CONTEXT_CHARS = 4000                                 # context length shown to the judge

# Supported file formats
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# --- Embedding models you can compare (all run locally on CPU, no API key) ------
ALLOWED_EMBEDDERS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2",
]

DEFAULT_STRATEGIES = [
    {"name": "Micro (256)", "chunk": 256, "overlap": 20},
    {"name": "Small (512)", "chunk": 512, "overlap": 50},
    {"name": "Large (1024)", "chunk": 1024, "overlap": 200},
]

STRATEGY_PRESETS = {
    "Standard": DEFAULT_STRATEGIES,
    "Fine-Grained": [
        {"name": "Tiny (128)", "chunk": 128, "overlap": 15},
        {"name": "Micro (256)", "chunk": 256, "overlap": 30},
        {"name": "Compact (384)", "chunk": 384, "overlap": 45},
    ],
    "Broad Context": [
        {"name": "Medium (512)", "chunk": 512, "overlap": 50},
        {"name": "Large (1024)", "chunk": 1024, "overlap": 150},
        {"name": "XL (2048)", "chunk": 2048, "overlap": 250},
    ],
}
