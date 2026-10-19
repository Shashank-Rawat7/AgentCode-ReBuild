import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Load the backend's environment file regardless of the
# directory from which the application is launched.
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"
REQUEST_TIMEOUT_MS = 60_000          # per request (HttpOptions.timeout is in milliseconds)
MAX_ATTEMPTS = 2                     # 1 initial attempt + 1 retry, never more
RETRY_DELAY_SECONDS = 2
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when Gemini fails or returns no usable text."""


class EmptyResponseError(LLMError):
    """Gemini answered (HTTP 200) but returned no usable text, even after the retry."""


def _describe_empty_response(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    finish_reasons = [str(getattr(c, "finish_reason", None)) for c in candidates]
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    block_message = getattr(feedback, "block_reason_message", None) if feedback else None
    return (
        "Gemini returned no text "
        f"(finish_reason={finish_reasons or None}, block_reason={block_reason}, "
        f"block_reason_message={block_message}, usage={getattr(response, 'usage_metadata', None)})"
    )


def chat(prompt: str) -> str:
    """Generate a text response using the Gemini API.

    Raises LLMError for API errors, timeouts and responses without usable text.
    Temporary failures (429, 5xx, timeout, empty response) are retried once.
    """
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_GENAI_API_KEY is not configured. "
            "Add it to backend/.env."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            # Retries are handled below; attempts=1 keeps the SDK from adding its own.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
        except errors.APIError as e:
            code = getattr(e, "code", None)
            last_error = LLMError(f"Gemini API error {code}: {getattr(e, 'message', None) or e}")
            retryable = code in RETRYABLE_STATUS_CODES
        except httpx.TimeoutException as e:
            last_error = LLMError(f"Gemini request timed out: {type(e).__name__}: {e}")
            retryable = True
        else:
            text = response.text
            if text and text.strip():
                return text.strip()
            last_error = EmptyResponseError(_describe_empty_response(response))
            retryable = True

        logger.warning("%s (attempt %d/%d)", last_error, attempt, MAX_ATTEMPTS)
        if not retryable or attempt == MAX_ATTEMPTS:
            break
        time.sleep(RETRY_DELAY_SECONDS)

    raise last_error
