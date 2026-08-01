# Crosspath AI Worker

Runpod Serverless worker using Ollama.

Supported tasks:

- `health`
- `resume_extract`
- `embed`
- `match_explanation`

The worker does not accept arbitrary browser-provided system prompts.

## Default models

- Generation: `qwen3:4b-instruct-2507-q4_K_M`
- Embeddings: `embeddinggemma`

Attach a Runpod network volume. Models are stored at:

```text
/runpod-volume/ollama-models
```

## Recommended Runpod settings

```text
Endpoint type: Queue
Active workers: 0
Max workers: 1
GPU: A4000 / A4500 / RTX 4000 (16 GB)
GPUs per worker: 1
Idle timeout: 10 seconds
Execution timeout: 900 seconds
Container disk: 20 GB
Network volume: 20 GB
FlashBoot: enabled
```

## Request URL

```text
POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync
Authorization: Bearer RUNPOD_API_KEY
Content-Type: application/json
```

The Runpod API key belongs only in server-side environment variables. Never use a `NEXT_PUBLIC_` variable for it.
