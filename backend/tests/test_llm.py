import logging

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from models import groq_llm
from models.groq_llm import EmptyResponseError, LLMError

# All tests here replace ChatGroq with a local fake: no network, no real API key.

FAKE_KEY = "gsk_test_key_never_logged"
REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def status_error(cls, code, message):
    return cls(message, response=httpx.Response(code, request=REQUEST), body=None)


def reply(content, finish_reason="stop", usage=None):
    extra = {}
    if usage:
        extra["usage_metadata"] = {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": usage[0] + usage[1]}
    return AIMessage(content=content, response_metadata={"finish_reason": finish_reason}, **extra)


def rate_limit(message, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    return groq.RateLimitError(message, response=httpx.Response(429, request=REQUEST, headers=headers), body=None)


class FakeChatGroq:
    """Stands in for langchain_groq.ChatGroq; returns or raises the queued outcomes in order.

    chat() builds a new ChatGroq for every call, so all fake clients in one test share
    a single queue: each outcome is consumed exactly once across calls.
    """

    def __init__(self, outcomes, **kwargs):
        self.outcomes = outcomes  # shared list, not a copy
        self.kwargs = kwargs
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def fake_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("AGENTCODE_MAX_PROMPT_TOKENS", raising=False)
    monkeypatch.delenv("AGENTCODE_MAX_429_WAIT_SECONDS", raising=False)
    sleeps = []
    monkeypatch.setattr(groq_llm.time, "sleep", lambda seconds: sleeps.append(seconds))
    created = []

    def install(*outcomes):
        queue = list(outcomes)

        def factory(**kwargs):
            client = FakeChatGroq(queue, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(groq_llm, "ChatGroq", factory)
        return created

    install.sleeps = sleeps
    return install


def test_success_returns_stripped_text_and_configures_client(fake_groq):
    created = fake_groq(reply("  hello  "))
    assert groq_llm.chat("hi") == "hello"
    client = created[0]
    assert client.calls == 1
    assert client.prompts == ["hi"]
    assert client.kwargs["api_key"] == FAKE_KEY
    assert client.kwargs["model"] == "openai/gpt-oss-20b"  # default when GROQ_MODEL is unset
    assert client.kwargs["timeout"] == groq_llm.REQUEST_TIMEOUT_SECONDS
    assert client.kwargs["max_retries"] == 0  # the SDK adds no retries of its own


def test_groq_model_env_overrides_default(fake_groq, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    created = fake_groq(reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    assert created[0].kwargs["model"] == "openai/gpt-oss-120b"


def test_gpt_oss_model_receives_explicit_output_cap_and_low_reasoning_effort(fake_groq):
    created = fake_groq(reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    client = created[0]
    assert client.kwargs["model"] == "openai/gpt-oss-20b"  # default; GPT-OSS family
    assert client.kwargs["max_tokens"] == groq_llm.MAX_OUTPUT_TOKENS
    assert client.kwargs["reasoning_effort"] == groq_llm.REASONING_EFFORT == "low"


def test_non_gpt_oss_model_does_not_receive_reasoning_effort(fake_groq, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    created = fake_groq(reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    client = created[0]
    assert client.kwargs["model"] == "llama-3.3-70b-versatile"
    assert client.kwargs["max_tokens"] == groq_llm.MAX_OUTPUT_TOKENS  # still applied, model-agnostic
    assert "reasoning_effort" not in client.kwargs  # GPT-OSS-only param, correctly withheld


def test_text_content_blocks_are_joined(fake_groq):
    fake_groq(AIMessage(content=[{"type": "text", "text": "[1, "}, {"type": "text", "text": "2]"}]))
    assert groq_llm.chat("hi") == "[1, 2]"


def test_empty_response_is_retried_once_then_succeeds(fake_groq):
    created = fake_groq(reply("", finish_reason="length"), reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    assert created[0].calls == 2


def test_empty_response_twice_raises_empty_response_error(fake_groq, caplog):
    created = fake_groq(reply("", finish_reason="length"), reply("   ", finish_reason="length"))
    with caplog.at_level(logging.WARNING, logger=groq_llm.logger.name):
        with pytest.raises(EmptyResponseError) as exc:
            groq_llm.chat("hi")
    assert "Groq returned no text" in str(exc.value)
    assert "finish_reason=length" in str(exc.value)
    assert created[0].calls == 2
    assert any("attempt 1/2" in r.getMessage() for r in caplog.records)


def test_server_error_is_retried_once(fake_groq):
    created = fake_groq(status_error(groq.InternalServerError, 503, "overloaded"), reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    assert created[0].calls == 2


def test_rate_limit_without_suggested_wait_is_not_retried(fake_groq):
    created = fake_groq(rate_limit("rate limit reached"), reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "429" in str(exc.value)
    assert "no suggested wait; not retried" in str(exc.value)
    assert not isinstance(exc.value, EmptyResponseError)
    assert created[0].calls == 1
    assert fake_groq.sleeps == []


def test_rate_limit_with_short_suggested_wait_is_retried_after_that_wait(fake_groq):
    created = fake_groq(
        rate_limit("Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7453, Requested 3533. Please try again in 1.5s."),
        reply("ok"),
    )
    assert groq_llm.chat("hi") == "ok"
    assert created[0].calls == 2
    assert fake_groq.sleeps == [1.5]


def test_rate_limit_retry_after_header_is_used(fake_groq):
    fake_groq(rate_limit("Rate limit reached.", retry_after="3"), reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    assert fake_groq.sleeps == [3.0]


def test_rate_limit_with_long_suggested_wait_fails_at_once(fake_groq):
    created = fake_groq(rate_limit("Rate limit reached. Please try again in 1m23.5s."), reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "suggested wait 83.5s exceeds 20s; not retried" in str(exc.value)
    assert created[0].calls == 1
    assert fake_groq.sleeps == []


def test_rate_limit_wait_cap_is_configurable(fake_groq, monkeypatch):
    monkeypatch.setenv("AGENTCODE_MAX_429_WAIT_SECONDS", "90")
    fake_groq(rate_limit("Please try again in 1m23.5s."), reply("ok"))
    assert groq_llm.chat("hi") == "ok"
    assert fake_groq.sleeps == [83.5]


def test_rate_limit_twice_is_reported_after_one_retry(fake_groq):
    created = fake_groq(rate_limit("Please try again in 2s."), rate_limit("Please try again in 2s."))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "retry already used" in str(exc.value)
    assert created[0].calls == 2


@pytest.mark.parametrize("text,expected", [
    ("Please try again in 7.66s.", 7.66),
    ("Please try again in 350ms.", 0.35),
    ("Please try again in 1m2s.", 62.0),
    ("Rate limit reached.", None),
])
def test_suggested_wait_parsing(text, expected):
    result = groq_llm.suggested_wait_seconds(rate_limit(text))
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_non_retryable_client_error_is_not_retried(fake_groq):
    created = fake_groq(status_error(groq.BadRequestError, 400, "bad request"), reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "400" in str(exc.value)
    assert created[0].calls == 1


def test_invalid_api_key_is_not_retried(fake_groq):
    created = fake_groq(status_error(groq.AuthenticationError, 401, "invalid api key"), reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "401" in str(exc.value)
    assert created[0].calls == 1


def test_timeout_is_retried_then_raises(fake_groq):
    created = fake_groq(groq.APITimeoutError(request=REQUEST), groq.APITimeoutError(request=REQUEST))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "timed out" in str(exc.value)
    assert created[0].calls == 2


def test_connection_error_is_reported_and_not_retried(fake_groq):
    created = fake_groq(groq.APIConnectionError(request=REQUEST), reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("hi")
    assert "Could not connect to Groq" in str(exc.value)
    assert created[0].calls == 1


def test_api_key_never_appears_in_errors_or_logs(fake_groq, caplog):
    fake_groq(rate_limit("slow down"), reply(""))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError) as exc:
            groq_llm.chat("hi")
    assert FAKE_KEY not in str(exc.value)
    assert FAKE_KEY not in caplog.text


def test_missing_api_key_raises_without_creating_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(groq_llm, "ChatGroq", lambda **kwargs: pytest.fail("client must not be created"))
    with pytest.raises(RuntimeError) as exc:
        groq_llm.chat("hi")
    assert "GROQ_API_KEY is not configured" in str(exc.value)


# --- Token usage measurement --------------------------------------------------
def test_usage_is_logged_from_the_groq_report(fake_groq, caplog):
    fake_groq(reply("ok", usage=(120, 30)))
    with caplog.at_level(logging.INFO, logger=groq_llm.logger.name):
        assert groq_llm.chat("hi") == "ok"
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM usage"))
    assert "model=openai/gpt-oss-20b" in line
    assert "input_tokens=120 output_tokens=30 total_tokens=150 source=groq-reported" in line
    assert FAKE_KEY not in caplog.text


def test_usage_falls_back_to_a_labelled_local_estimate(fake_groq, caplog):
    fake_groq(reply("abcdefgh"))  # no usage_metadata from the provider
    with caplog.at_level(logging.INFO, logger=groq_llm.logger.name):
        groq_llm.chat("x" * 40)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM usage"))
    assert "input_tokens=10 output_tokens=2 total_tokens=12 source=local-estimate" in line


def test_task_usage_totals_every_call_including_empty_replies(fake_groq):
    fake_groq(reply("", usage=(50, 40)), reply("ok", usage=(60, 10)), reply("again", usage=(5, 5)))
    token = groq_llm.start_task_usage()
    try:
        assert groq_llm.chat("first") == "ok"   # empty reply, then retry: both count
        assert groq_llm.chat("second") == "again"
    finally:
        summary = groq_llm.finish_task_usage(token)  # always reset, even if an assert fails
    assert summary == {
        "calls": 3, "input_tokens": 115, "output_tokens": 55, "total_tokens": 170,
        "groq_reported_calls": 3, "estimated_calls": 0,
    }


def test_usage_outside_a_task_is_logged_but_not_collected(fake_groq):
    fake_groq(reply("ok", usage=(1, 1)))
    assert groq_llm._task_usage.get() is None
    groq_llm.chat("hi")
    assert groq_llm._task_usage.get() is None


def test_purpose_is_the_langgraph_node_name(fake_groq, caplog):
    from typing import TypedDict
    from langgraph.graph import END, StateGraph

    class State(TypedDict, total=False):
        answer: str

    def my_planner_node(state):
        return {"answer": groq_llm.chat("hi")}

    builder = StateGraph(State)
    builder.add_node("my_planner_node", my_planner_node)
    builder.set_entry_point("my_planner_node")
    builder.add_edge("my_planner_node", END)
    fake_groq(reply("ok", usage=(3, 1)))
    with caplog.at_level(logging.INFO, logger=groq_llm.logger.name):
        assert builder.compile().invoke({"answer": ""}) == {"answer": "ok"}
    assert any("purpose=my_planner_node" in r.getMessage() for r in caplog.records)


# --- Optional prompt-size guard ---------------------------------------------------
def test_prompt_guard_is_disabled_by_default(fake_groq):
    created = fake_groq(reply("ok"))
    assert groq_llm.chat("x" * 200_000) == "ok"
    assert created[0].calls == 1


def test_prompt_guard_blocks_oversized_prompt_when_configured(fake_groq, monkeypatch):
    monkeypatch.setenv("AGENTCODE_MAX_PROMPT_TOKENS", "100")
    created = fake_groq(reply("unused"))
    with pytest.raises(LLMError) as exc:
        groq_llm.chat("x" * 1000)
    assert "about 250 tokens (local estimate)" in str(exc.value)
    assert "not sent to Groq" in str(exc.value)
    assert created == []  # no client, no request


# --- Optional LangSmith tracing ---------------------------------------------------
def test_tracing_is_off_by_default_and_not_required(fake_groq, monkeypatch):
    from pathlib import Path
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    example = (Path(groq_llm.__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "LANGSMITH_TRACING=false" in example
    assert "LANGSMITH_API_KEY=\n" in example.replace("\r\n", "\n")  # no key value shipped
    fake_groq(reply("ok"))
    assert groq_llm.chat("hi") == "ok"  # works with no LangSmith configuration at all
