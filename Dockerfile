FROM ollama/ollama:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/runpod-volume/ollama-models
ENV OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M
ENV OLLAMA_EMBED_MODEL=embeddinggemma
ENV MAX_RESUME_CHARS=50000

CMD ["python3", "-u", "/app/handler.py"]
