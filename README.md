# 🧪 RAG Evaluation & Optimization Tool

Find out **which RAG settings work best on your own documents**. Upload any document (PDF, TXT, Markdown), ask questions, and the tool runs experiments across combinations of **chunk sizes, overlap, and embedding models** in parallel. It answers your questions with each combination, while an impartial **LLM judge** scores every answer for relevance and factual faithfulness (hallucination detection).

Built entirely on **FastAPI** with a modern, responsive web dashboard (Tailwind CSS, Plotly.js, Lucide Icons) served directly on a single port.

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v2.0_Unified_App-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Groq](https://img.shields.io/badge/Groq-Ultra--Fast_Inference-f55036)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 💡 Why This Tool?
Engineering teams often pick arbitrary defaults like `chunk_size=1024` or an embedding model without empirical validation. RAG pipelines can degrade due to **retrieval failures** (missing context) or **generation failures** (hallucinations or poor reasoning). Manual testing across combinations is slow and subjective.

This tool automates scientific comparisons so you can choose chunk sizes, overlap, and embedding models with hard evidence.

---

## ✨ Features
- ⚡ **1-Click Sample Benchmark:** Test the entire platform immediately with pre-loaded RAG documentation and evaluation questions—no file upload required!
- 📄 **Multi-Format Ingestion:** Full native support for **PDF, TXT, and Markdown (.md)** files.
- 🧩 **Customizable Chunk Strategies:** Editable interactive table with presets (*Standard*, *Fine-Grained*, *Broad Context*) or custom chunk size and overlap.
- 🧠 **Local Embedding Models:** Compare top sentence-transformers (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `sentence-transformers/all-mpnet-base-v2`, `paraphrase-MiniLM-L3-v2`) locally on CPU with **zero API cost**.
- ⚖️ **Impartial LLM Judge:** Evaluates both **Relevance** (answering the question) and **Faithfulness** (strictly grounded in retrieved context; penalizing hallucinations).
- 🔍 **Retrieval Hit-Rate Verification:** Specify expected phrases using `question || expected text` to verify if the retriever actually fetched the required context.
- 📊 **Rich Visual Analytics:**
  - **Radar / Spider Chart:** Multi-dimensional balance across Relevance, Faithfulness, Hit-Rate, and Speed.
  - **Pareto Tradeoff Scatter Plot:** Latency vs. Quality score with chunk count bubble sizing.
  - **Side-by-Side Question Inspector:** Compare retrieved chunks, answers, and judge rationales side-by-side.
- 📥 **Executive Reporting:** Download raw **CSV**, full **JSON**, and a formatted **Executive Markdown Report** ready to share with your team or paste into PRs.
- 🚀 **Parallel & Isolated Execution:** Parallelized with async worker pools; each experiment runs in an isolated ephemeral ChromaDB collection.
- ⚡ **Pure FastAPI Architecture:** Single unified application serving both the high-performance REST API and the modern interactive web dashboard on Port 8000.

---

## 🏗️ Architecture

```
User Document (.pdf, .txt, .md) + Questions (with optional expected phrase)
                             │
            FastAPI Unified Server (Port 8000)
    ┌────────────────────────┴────────────────────────┐
    │                                                 │
    ▼                                                 ▼
Modern Web UI (Tailwind + Plotly)              REST API Endpoints
(/, /static)                                   (/optimize, /config, /sample)
                                                      │
           ┌──────────────────────────────────────────┘
           │
           ├─► [Strategy A × Embedder 1] (Isolated Chroma Collection)
           ├─► [Strategy B × Embedder 2] (Isolated Chroma Collection)
           │     └─► Split -> Embed -> Retrieve Top-K -> Groq LLM Answer -> LLM Judge Score
           │
           ▼
  Aggregated Performance Metrics (Relevance, Faithfulness, Hit-Rate, Latency, Token Usage)
           │
           ▼
  Leaderboard Table, Radar Chart, Pareto Frontier, Side-by-Side Inspector, Markdown/CSV Export
```

---

## 🛠️ Tech Stack
- **Unified Web & API Framework:** FastAPI (serves both REST API and modern static UI).
- **Frontend Dashboard:** Modern Vanilla JS, Tailwind CSS, Plotly.js, Lucide Icons.
- **RAG Engine:** LangChain, ChromaDB, HuggingFace Sentence-Transformers, Groq API.
- **Testing & CI:** Pytest, HTTPX, GitHub Actions CI.
- **Containerization:** Docker Compose with volume-cached Hugging Face models.

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository and navigate to the project
```bash
git clone https://github.com/your-username/rag-evaluation-tool.git
cd rag-evaluation-tool
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env` and insert your free Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```
> Get a free Groq API key in seconds at [console.groq.com](https://console.groq.com).

### 5. Launch the Application (Single Command!)
```bash
uvicorn backend.main:app --reload --port 8000
```
- **Web Dashboard:** Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive API Docs (Swagger):** Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Docker Setup
Run the unified container with a single command:
```bash
cp .env.example .env     # Ensure your GROQ_API_KEY is inside
docker compose up --build
```
Open **http://localhost:8000** in your browser.

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | - | Groq API key for generator and judge models |
| `GENERATOR_MODEL` | No | `llama-3.3-70b-versatile` | Model used to generate RAG answers |
| `JUDGE_MODEL` | No | `openai/gpt-oss-120b` | Model used to score answers |
| `TOP_K` | No | `4` | Number of chunks retrieved per question |
| `MAX_PARALLEL` | No | `3` | Maximum concurrent experiments |
| `MAX_UPLOAD_MB` | No | `20` | Maximum file upload size in MB |
| `HF_TOKEN` | No | - | Optional HuggingFace token for higher download rate |

---

## 🧪 Running Automated Tests
The test suite validates helper parsers, resilient JSON extraction, clamping logic, API routes, and input validations.

```bash
python -m pytest
```

---

## 📊 Evaluation Metrics Explained

1. **Relevance Score (1 to 10):**
   - Assesses whether the system answer directly and clearly addresses the question.
2. **Faithfulness Score (1 to 10):**
   - Evaluates factual consistency against the retrieved text. Claims invented outside the context are penalized (hallucination detection).
3. **Retrieval Hit-Rate (%):**
   - Measures whether the retriever successfully retrieved the expected ground-truth passage or phrase.
4. **Latency (seconds):**
   - Average wall-clock time required for retrieval and generation per question.
5. **Token Cost:**
   - Input and output token counts tracked per experiment.

---

## 🤝 Contributing & License
Contributions, feedback, and issues are welcome! Feel free to open a PR or submit an issue.

Licensed under the [MIT License](LICENSE).
