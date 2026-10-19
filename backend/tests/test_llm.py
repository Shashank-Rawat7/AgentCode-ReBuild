import logging
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from models import groq_llm
from models.groq_llm import LLMError


class FakeResponse:
    def __init__(self, text=None, finish_reason=None, block_reason=None):
        self.text = text
        self.candidates = [SimpleNamespace(finish_reason=finish_reason)] if finish_reason else []
        self.prompt_feedback = (
            SimpleNamespace(block_reason=block_reason, block_reason_message=None) if block_reason else None
        )
        self.usage_metadata = None


class FakeClient:
    """Stands in for genai.Client; returns or raises the queued outcomes in order."""

    def __init__(self, outcomes, **kwargs):
        self.outcomes = list(outcomes)
        self.kwargs = kwargs
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key")
    monkeypatch.setattr(groq_llm.time, "sleep", lambda seconds: None)
    created = []

    def install(*outcomes):
        def factory(**kwargs):
            client = FakeClient(outcomes, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(groq_llm.genai, "Client", factory)
        return created

    return install


def api_error(cls, code, message):
    return cls(code, {"error": {"code": code, "message": message, "status": "TEST"}})


def test_success_returns_stripped_text_and_sets_timeout(fake_gemini):
    created = fake_gemini(FakeResponse(text="  hello  "))
    assert groq_llm.chat("hi") == "hello"
    client = created[0]
    assert client.calls == 1
    assert client.kwargs["http_options"].timeout == groq_llm.REQUEST_TIMEOUT_MS
    assert client.kwargs["http_options"].retry_options.attempts == 1  # SDK adds no retries of its own


def test_empty_response_is_retried_once_then_succeeds(fake_gemini):
    created = fake_gemini(FakeResponse(text=None, finish_reason="OTHER"), FakeResponse(text="ok"))
    assert groq_llm.chat("hi") == "ok"
    assert created[0].calls == 2


def test_empty_response_twice_raises_with_metadata(fake_gemini, caplog):
    created = fake_gemini(
        FakeResponse(text=None, finish_reason="SAFETY"),
        FakeResponse(text=None, finish_reason="SAFETY"),
    )
    with caplog.at_level(logging.WARNING, logger=groq_llm.logger.name):
        with pytest.raises(LLMError) as exc:
            groq_llm.chat("hi")
    assert "Gemini returned no text" in str(exc.value)
    assert "SAFETY" in str(exc.value)
    assert created[0].calls == 2
    assert any("attempt 1/2" in r.getMessage() for r in caplog.records)


def test_blank_text_counts_as_empty(fake_gemini):
    created = fake_gemini(FakeResponse(text="   "), FakeResponse(text=""))
    with pytest.raises(LLMError):
        groq_llm.chat("hi")
    assert created[0].calls == 2


def test_blocked_prompt_reason_is_reported(fake_gemini):
    fake_gemini(FakeResponse(block_reason="PROHIBITED_CONTENT"), FakeResponse(block_reason="PROHIBITED_CONTENT"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "PROHIBITED_CONTENT" in str(exc.value)


def test_server_error_is_retried_once(fake_gemini):
    created = fake_gemini(api_error(errors.ServerError, 503, "overloaded"), FakeResponse(text="ok"))
    assert groq_llm.chat("hi") == "ok"
    assert created[0].calls == 2


def test_rate_limit_twice_raises_with_code(fake_gemini):
    created = fake_gemini(
        api_error(errors.ClientError, 429, "quota"),
        api_error(errors.ClientError, 429, "quota"),
    )
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "429" in str(exc.value)
    assert created[0].calls == 2


def test_non_retryable_client_error_is_not_retried(fake_gemini):
    created = fake_gemini(api_error(errors.ClientError, 400, "bad request"), FakeResponse(text="unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "400" in str(exc.value)
    assert created[0].calls == 1


def test_timeout_is_retried_then_raises(fake_gemini):
    created = fake_gemini(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "timed out" in str(exc.value)
    assert created[0].calls == 2


def test_missing_api_key_raises_without_creating_client(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setattr(groq_llm.genai, "Client", lambda **kwargs: pytest.fail("client must not be created"))
    with pytest.raises(RuntimeError) as exc:
        groq_llm.chat("hi")
    assert "GOOGLE_GENAI_API_KEY is not configured" in str(exc.value)
