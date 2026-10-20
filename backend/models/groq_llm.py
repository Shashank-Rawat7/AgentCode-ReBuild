import contextvars
import logging
import os
import re
import time
from pathlib import Path

import groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the backend's environment file regardless of the
# directory from which the application is launched.
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"   # override with GROQ_MODEL in backend/.env
REQUEST_TIMEOUT_SECONDS = 60           # per request
MAX_ATTEMPTS = 2                       # 1 initial attempt + 1 retry, never more
RETRY_DELAY_SECONDS = 2
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}   # 429 is handled separately below
DEFAULT_MAX_429_WAIT_SECONDS = 20      # longer suggested waits fail at once
CHARS_PER_TOKEN_ESTIMATE = 4           # rough local estimate, used only when Groq reports nothing


class LLMError(RuntimeError):
    """Raised when the Groq model fails or returns no usable text."""


class EmptyResponseError(LLMError):
    """Groq answered but returned no usable text, even after the retry."""


# --- Token usage tracking -------------------------------------------------------
# One list per /run-task request. The list object is shared (never reassigned), so
# LLM calls made anywhere inside the workflow add to the same task total.
_task_usage = contextvars.ContextVar("agentcode_task_usage", default=None)


def start_task_usage():
    """Begin collecting LLM usage for one task. Returns a token for finish_task_usage()."""
    return _task_usage.set([])


def finish_task_usage(token) -> dict:
    """Stop collecting and return the task's usage summary."""
    records = _task_usage.get() or []
    _task_usage.reset(token)
    return summarize_usage(records)


def summarize_usage(records: list) -> dict:
    return {
        "calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "total_tokens": sum(r["total_tokens"] for r in records),
        "groq_reported_calls": sum(1 for r in records if r["source"] == "groq-reported"),
        "estimated_calls": sum(1 for r in records if r["source"] == "local-estimate"),
    }


def estimate_tokens(text: str) -> int:
    return len(text or "") // CHARS_PER_TOKEN_ESTIMATE


def _current_purpose() -> str:
    """Name of the LangGraph node making the call, when called inside a graph."""
    try:
        from langgraph.config import get_config
        return (get_config().get("metadata") or {}).get("langgraph_node") or "unknown"
    except Exception:
        return "unknown"


def _record_usage(message, prompt: str, text: str, model: str, purpose: str) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage.get("total_tokens") is not None:
        record = {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage["total_tokens"]),
            "source": "groq-reported",
        }
    else:
        prompt_est, output_est = estimate_tokens(prompt), estimate_tokens(text)
        record = {
            "input_tokens": prompt_est,
            "output_tokens": output_est,
            "total_tokens": prompt_est + output_est,
            "source": "local-estimate",
        }
    record.update(purpose=purpose, model=model)
    logger.info(
        "LLM usage purpose=%s model=%s input_tokens=%d output_tokens=%d total_tokens=%d source=%s",
        purpose, model, record["input_tokens"], record["output_tokens"], record["total_tokens"], record["source"],
    )
    records = _task_usage.get()
    if records is not None:
        records.append(record)


# --- Configuration ----------------------------------------------------------------
def get_model_name() -> str:
    return os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def get_max_prompt_tokens() -> int:
    """Optional guard on the estimated prompt size. 0 (the default) disables it."""
    return _int_env("AGENTCODE_MAX_PROMPT_TOKENS", 0)


def get_max_429_wait_seconds() -> int:
    return _int_env("AGENTCODE_MAX_429_WAIT_SECONDS", DEFAULT_MAX_429_WAIT_SECONDS)


def suggested_wait_seconds(error) -> float | None:
    """Wait suggested by a 429: the retry-after header, or Groq's 'try again in ...' text."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in\s+(?:(\d+)m)?(?:([\d.]+)s|([\d.]+)ms)", str(getattr(error, "message", "") or error))
    if not match:
        return None
    minutes, seconds, millis = match.groups()
    total = int(minutes or 0) * 60
    if seconds:
        total += float(seconds)
    if millis:
        total += float(millis) / 1000
    return total


def _text_of(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks: keep only the text parts
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        )
    return ""


def _describe_empty_response(message, model: str) -> str:
    metadata = getattr(message, "response_metadata", None) or {}
    return (
        "Groq returned no text "
        f"(model={model}, finish_reason={metadata.get('finish_reason')}, "
        f"usage={getattr(message, 'usage_metadata', None)})"
    )


def chat(prompt: str) -> str:
    """Generate a text response with a Groq-hosted model.

    Raises LLMError for API errors, timeouts and responses without usable text.
    5xx errors, timeouts and empty responses are retried once. A 429 is retried
    once only when Groq suggests a short wait; otherwise it fails at once.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to backend/.env."
        )

    model = get_model_name()
    purpose = _current_purpose()

    max_prompt_tokens = get_max_prompt_tokens()
    if max_prompt_tokens > 0:
        estimate = estimate_tokens(prompt)
        if estimate > max_prompt_tokens:
            raise LLMError(
                f"Prompt for {purpose} is about {estimate} tokens (local estimate), above "
                f"AGENTCODE_MAX_PROMPT_TOKENS={max_prompt_tokens}; not sent to Groq"
            )

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,  # retries are handled below, so the SDK adds none of its own
    )

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        delay = RETRY_DELAY_SECONDS
        try:
            message = llm.invoke(prompt)
        except groq.RateLimitError as e:
            wait = suggested_wait_seconds(e)
            limit = get_max_429_wait_seconds()
            retryable = wait is not None and wait <= limit
            if wait is None:
                note = "no suggested wait; not retried"
            elif not retryable:
                note = f"suggested wait {wait:g}s exceeds {limit}s; not retried"
            elif attempt < MAX_ATTEMPTS:
                note = f"retrying after the suggested {wait:g}s"
            else:
                note = "retry already used"
            last_error = LLMError(f"Groq rate limit (429) during {purpose}: {e.message} [{note}]")
            delay = wait if retryable else 0
        except groq.APIStatusError as e:
            code = e.status_code
            last_error = LLMError(f"Groq API error {code}: {e.message}")
            retryable = code in RETRYABLE_STATUS_CODES
        except groq.APITimeoutError as e:
            last_error = LLMError(f"Groq request timed out: {type(e).__name__}: {e}")
            retryable = True
        except groq.APIConnectionError as e:
            last_error = LLMError(f"Could not connect to Groq: {type(e).__name__}: {e}")
            retryable = False
        else:
            text = _text_of(message)
            _record_usage(message, prompt, text, model, purpose)  # empty replies still use tokens
            if text and text.strip():
                return text.strip()
            last_error = EmptyResponseError(_describe_empty_response(message, model))
            retryable = True

        logger.warning("%s (attempt %d/%d)", last_error, attempt, MAX_ATTEMPTS)
        if not retryable or attempt == MAX_ATTEMPTS:
            break
        time.sleep(delay)

    raise last_error
