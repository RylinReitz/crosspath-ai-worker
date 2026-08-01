# Crosspath AI Worker v3

Runpod Serverless worker using Ollama.

## Résumé pipeline

The `resume_extract` task now has three stages:

1. The model extracts factual fields and copies role date text.
2. Python normalizes missing values, recovers date ranges, merges overlapping
   employment intervals, and calculates experience.
3. The model writes a summary from validated JSON only. Invalid summary output
   is replaced by a deterministic fallback.

The response includes `validation.member_review_required: true`. Crosspath must
show the generated profile to the member before using it for matching.

## Supported tasks

- `health`
- `resume_extract`
- `embed`
- `match_explanation`

## Local deterministic tests

```bash
python3 -m unittest -v test_resume_pipeline.py
```

## Runpod settings

```text
Endpoint type: Queue
Active workers: 0
Max workers: 1
GPU count: 1
Idle timeout: 5 seconds
Execution timeout: 900 seconds
Network volume: mounted at /runpod-volume
FlashBoot: enabled
```

## Request URL

```text
POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync
Authorization: Bearer RUNPOD_API_KEY
Content-Type: application/json
```

Keep the API key on the Crosspath server only. Never expose it through a
`NEXT_PUBLIC_` environment variable.
