"""
llm_report.py
-------------
Generates a plain-language explanation of a skin-lesion classifier's
output using the HuggingFace Inference API (Qwen2.5-7B-Instruct).

Design decisions
----------------
* Stateless function interface — no global client state so callers can
  inject their own token or swap the model without touching main.py.
* Hard JSON-only system prompt + defensive fence-stripping so the
  endpoint never 500s due to LLM formatting non-compliance.
* Exponential backoff on 429 (free-tier rate limit) — up to 3 retries
  with 2 s / 4 s / 8 s delays before giving up and returning the
  graceful fallback dict.
* Temperature 0.3 — narrating fixed numbers, not creative writing.
"""

import json
import logging
import time
from typing import Optional

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model & prompt config
# ---------------------------------------------------------------------------

_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_MAX_TOKENS = 400
_TEMPERATURE = 0.3
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   # seconds; doubles on each retry

_SYSTEM_PROMPT = """You are explaining a skin lesion classifier's output to a non-expert user.
You are NOT diagnosing — you are explaining what the model output means and how confident it is.

Rules:
- If `inconclusive` is true, you MUST clearly state the result is inconclusive and explain why
  (below threshold, or too close between top classes).
- Never state a class as certain if inconclusive is true.
- Always include a disclaimer that this is not a medical diagnosis and to consult a dermatologist.
- Respond ONLY with valid JSON matching the schema below. No markdown, no preamble, no code fences."""

_RESPONSE_SCHEMA = """{
  "headline": "one-line plain-language summary",
  "explanation": "2-4 sentences explaining the result, referencing relevant thresholds in plain terms",
  "confidence_level": "high" | "moderate" | "low" | "inconclusive",
  "inconclusive": true or false,
  "disclaimer": "short standard disclaimer"
}"""

_FALLBACK_DISCLAIMER = (
    "This is not a medical diagnosis. Please consult a qualified dermatologist."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_user_prompt(probs: dict, thresholds: dict, eval_result: dict) -> str:
    return (
        f"Class probabilities: {json.dumps(probs)}\n"
        f"Per-class thresholds: {json.dumps(thresholds)}\n"
        f"Evaluation: {json.dumps(eval_result)}\n\n"
        f"Return JSON exactly in this shape:\n{_RESPONSE_SCHEMA}"
    )


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_report(
    probs: dict,
    thresholds: dict,
    eval_result: dict,
    hf_token: str,
    model: str = _MODEL,
) -> dict:
    """
    Call the HF Inference API to generate a plain-language report.

    Parameters
    ----------
    probs       : {label: float}  — softmax probabilities from the classifier
    thresholds  : {label: float}  — per-class decision thresholds
    eval_result : dict            — output of predict_calibrated() in main.py
    hf_token    : str             — HuggingFace API token (from env / Space secret)
    model       : str             — HF model ID (override for testing)

    Returns
    -------
    dict matching the JSON schema above, or a graceful fallback dict on any
    unrecoverable error so the /predict endpoint never 500s due to LLM issues.
    """
    client = InferenceClient(model=model, token=hf_token)
    user_prompt = _build_user_prompt(probs, thresholds, eval_result)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
            raw_text = response.choices[0].message.content.strip()

            # Defensive stripping: smaller instruct models often ignore
            # "no code fences" and wrap output in ```json … ``` anyway.
            raw_text = (
                raw_text
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )

            return json.loads(raw_text)

        except json.JSONDecodeError as exc:
            # LLM returned non-JSON — not worth retrying (it's a prompt issue)
            logger.warning("LLM returned non-JSON output: %s", exc)
            last_error = exc
            break

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = "429" in err_str or "rate limit" in err_str

            if is_rate_limit and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.0f s …",
                    attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                last_error = exc
                continue

            logger.error("LLM report generation failed: %s", exc)
            last_error = exc
            break

    # ── Graceful fallback ────────────────────────────────────────────────────
    logger.error("Returning fallback report after error: %s", last_error)
    return {
        "headline": "Unable to generate explanation",
        "explanation": (
            "The report could not be generated automatically. "
            "Please review the raw probabilities below."
        ),
        "confidence_level": eval_result.get("confidence_level", "unknown"),
        "inconclusive":     eval_result.get("inconclusive", True),
        "disclaimer":       _FALLBACK_DISCLAIMER,
    }
