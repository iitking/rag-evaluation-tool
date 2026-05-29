FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000
# PORT can be overridden by the host (e.g. 7860 on Hugging Face Spaces)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
