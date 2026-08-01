import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

import requests
import runpod

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATION_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct-2507-q4_K_M",
)
EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "embeddinggemma",
)
MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "50000"))

MODEL_LOCK = threading.Lock()
READY_MODELS: set[str] = set()

RESUME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "professional_summary": {"type": "string"},
        "current_or_most_recent_role": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "organization": {"type": "string"},
                "location": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": [
                "title",
                "organization",
                "location",
                "start_date",
                "end_date",
            ],
            "additionalProperties": False,
        },
        "work_history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "organization": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "organization",
                    "location",
                    "start_date",
                    "end_date",
                    "highlights",
                ],
                "additionalProperties": False,
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "graduation_year": {"type": "string"},
                },
                "required": [
                    "institution",
                    "degree",
                    "field",
                    "graduation_year",
                ],
                "additionalProperties": False,
            },
        },
        "industries": {"type": "array", "items": {"type": "string"}},
        "professional_functions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "career_topics": {
            "type": "array",
            "items": {"type": "string"},
        },
        "years_experience_estimate": {"type": "number"},
        "can_advise_on": {
            "type": "array",
            "items": {"type": "string"},
        },
        "possible_learning_interests": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "professional_summary",
        "current_or_most_recent_role",
        "work_history",
        "education",
        "industries",
        "professional_functions",
        "skills",
        "career_topics",
        "years_experience_estimate",
        "can_advise_on",
        "possible_learning_interests",
        "evidence_notes",
    ],
    "additionalProperties": False,
}

MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "explanation": {"type": "string"},
        "shared_topics": {"type": "array", "items": {"type": "string"}},
        "complementary_topics": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "headline",
        "explanation",
        "shared_topics",
        "complementary_topics",
    ],
    "additionalProperties": False,
}


def start_ollama() -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    env.setdefault("OLLAMA_MODELS", "/runpod-volume/ollama-models")

    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("Ollama stopped during startup.")

        try:
            response = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=2,
            )
            if response.ok:
                return process
        except requests.RequestException:
            pass

        time.sleep(1)

    process.terminate()
    raise RuntimeError("Ollama did not become ready in time.")


OLLAMA_PROCESS = start_ollama()


def model_is_available(model: str) -> bool:
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=15,
    )
    response.raise_for_status()
    names = {
        item.get("name")
        for item in response.json().get("models", [])
    }
    return model in names


def ensure_model(model: str) -> None:
    if model in READY_MODELS:
        return

    with MODEL_LOCK:
        if model in READY_MODELS:
            return

        if not model_is_available(model):
            print(f"Pulling Ollama model: {model}", flush=True)
            result = subprocess.run(
                ["ollama", "pull", model],
                check=False,
                text=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Unable to pull Ollama model: {model}"
                )

        READY_MODELS.add(model)


def ollama_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    ensure_model(GENERATION_MODEL)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_ctx": 16384,
            },
        },
        timeout=(15, 600),
    )
    response.raise_for_status()

    payload = response.json()
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The model returned invalid structured JSON."
        ) from error

    return {
        "result": parsed,
        "model": payload.get("model", GENERATION_MODEL),
        "usage": {
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
            "total_duration_ns": payload.get("total_duration"),
            "load_duration_ns": payload.get("load_duration"),
        },
    }


def extract_resume(resume_text: str) -> dict[str, Any]:
    cleaned = resume_text.strip()

    if len(cleaned) < 100:
        raise ValueError("The resume text is too short to analyze.")

    if len(cleaned) > MAX_RESUME_CHARS:
        raise ValueError(
            f"The resume exceeds the {MAX_RESUME_CHARS:,}-character limit."
        )

    system_prompt = """
You extract professional information from resumes for Crosspath.

Use only information explicitly stated in the resume. Do not invent
employers, dates, degrees, skills, accomplishments, seniority, or goals.
Use empty strings or empty arrays when the source does not provide a
field. Phrase possible learning interests cautiously and only when they
follow directly from the documented career path.

Do not extract or infer age, date of birth, race, ethnicity, religion,
gender, disability, marital status, citizenship, political beliefs, or
other sensitive characteristics. Do not include phone numbers, email
addresses, street addresses, photographs, references, or contact data.

Write the summary in the third person without using the person's name.
Keep it factual and concise. Put uncertainty in evidence_notes. Return
only JSON matching the supplied schema.
""".strip()

    return ollama_chat(
        system_prompt=system_prompt,
        user_prompt=(
            "Extract a structured professional profile from this resume:\n\n"
            f"{cleaned}"
        ),
        schema=RESUME_SCHEMA,
    )


def create_embedding(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text is required for an embedding.")
    if len(cleaned) > 20000:
        raise ValueError(
            "Embedding text exceeds the 20,000-character limit."
        )

    ensure_model(EMBEDDING_MODEL)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": cleaned,
            "truncate": True,
            "keep_alive": "10m",
        },
        timeout=(15, 300),
    )
    response.raise_for_status()

    payload = response.json()
    embeddings = payload.get("embeddings")
    if (
        not isinstance(embeddings, list)
        or not embeddings
        or not isinstance(embeddings[0], list)
    ):
        raise RuntimeError(
            "The embedding model returned an invalid vector."
        )

    return {
        "embedding": embeddings[0],
        "dimensions": len(embeddings[0]),
        "model": payload.get("model", EMBEDDING_MODEL),
    }


def explain_match(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = """
You explain professional matches on Crosspath.

Use only the two approved professional profiles. Explain concrete
professional overlap or complementarity. Do not infer sensitive traits,
personality, friendship potential, prestige, wealth, intelligence, or
cultural fit. Do not claim that either person wants something unless the
profile states it. Keep the explanation under 90 words. Return only JSON
matching the supplied schema.
""".strip()

    return ollama_chat(
        system_prompt=system_prompt,
        user_prompt=(
            "Profile A:\n"
            f"{json.dumps(profile_a, ensure_ascii=False)}\n\n"
            "Profile B:\n"
            f"{json.dumps(profile_b, ensure_ascii=False)}"
        ),
        schema=MATCH_SCHEMA,
    )


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = event.get("input") or {}
        if not isinstance(job_input, dict):
            raise ValueError("The job input must be an object.")

        task = job_input.get("task")

        if task == "health":
            return {
                "ok": True,
                "generation_model": GENERATION_MODEL,
                "embedding_model": EMBEDDING_MODEL,
            }

        if task == "resume_extract":
            resume_text = job_input.get("resume_text")
            if not isinstance(resume_text, str):
                raise ValueError(
                    "resume_text must be provided as a string."
                )
            return {"ok": True, **extract_resume(resume_text)}

        if task == "embed":
            text = job_input.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be provided as a string.")
            return {"ok": True, **create_embedding(text)}

        if task == "match_explanation":
            profile_a = job_input.get("profile_a")
            profile_b = job_input.get("profile_b")
            if not isinstance(profile_a, dict):
                raise ValueError("profile_a must be an object.")
            if not isinstance(profile_b, dict):
                raise ValueError("profile_b must be an object.")
            return {
                "ok": True,
                **explain_match(profile_a, profile_b),
            }

        raise ValueError(
            "Unknown task. Supported tasks are health, resume_extract, "
            "embed, and match_explanation."
        )

    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "error_type": "validation_error",
        }
    except requests.RequestException as error:
        print(f"Ollama request failed: {error}", file=sys.stderr)
        return {
            "ok": False,
            "error": "The AI service could not complete the request.",
            "error_type": "upstream_error",
        }
    except Exception as error:
        print(f"Unhandled worker error: {error}", file=sys.stderr)
        return {
            "ok": False,
            "error": "The AI worker encountered an unexpected error.",
            "error_type": "worker_error",
        }


runpod.serverless.start({"handler": handler})
