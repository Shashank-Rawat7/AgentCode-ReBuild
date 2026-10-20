import logging

import pytest
from fastapi.testclient import TestClient

import main
from langgraph_app import graph as wf


def run_workflow(task="create a file"):
    return wf.setup_graph().invoke({"task": task})


def developer_must_not_run(steps):
    pytest.fail("developer_node must not run when planning failed")


# --- graph: planner failures ---------------------------------------------------
def test_planner_exception_is_logged_with_traceback_and_reported(monkeypatch, caplog):
    def boom(task):
        raise RuntimeError("boom")

    monkeypatch.setattr(wf, "planner_agent", boom)
    monkeypatch.setattr(wf, "developer_agent", developer_must_not_run)

    with caplog.at_level(logging.ERROR, logger=wf.logger.name):
        result = run_workflow()

    assert result["status"] == "error"
    assert result["stage"] == "planner"
    assert result["error"] == "RuntimeError: boom"  # original error, not "No steps found"
    records = [r for r in caplog.records if "Planner failed" in r.getMessage()]
    assert records, "planner exception was not logged"
    assert records[0].exc_info is not None  # traceback attached


def test_planner_with_no_steps_is_reported_and_developer_not_run(monkeypatch):
    monkeypatch.setattr(wf, "planner_agent", lambda task: [])
    monkeypatch.setattr(wf, "developer_agent", developer_must_not_run)

    result = run_workflow()

    assert result["status"] == "error"
    assert result["stage"] == "planner"
    assert result["error"] == "Planner produced no steps"


# --- graph: developer outcomes ----------------------------------------------------
STEPS = [{"file": "a.py", "tool": "write", "args": {"content": "x"}}]


def test_failed_developer_step_is_reported(monkeypatch):
    monkeypatch.setattr(wf, "planner_agent", lambda task: STEPS)
    monkeypatch.setattr(
        wf, "developer_agent",
        lambda steps: ["[ERROR] write on a.py failed: Path is outside the workspace: a.py"],
    )

    result = run_workflow()

    assert result["status"] == "error"
    assert result["stage"] == "developer"
    assert "1 of 1 step(s) failed" in result["error"]
    assert "a.py" in result["error"]


def test_developer_exception_is_reported(monkeypatch):
    def crash(steps):
        raise ValueError("graph crashed")

    monkeypatch.setattr(wf, "planner_agent", lambda task: STEPS)
    monkeypatch.setattr(wf, "developer_agent", crash)

    result = run_workflow()

    assert result["status"] == "error"
    assert result["stage"] == "developer"
    assert result["error"] == "ValueError: graph crashed"


def test_successful_workflow_is_complete(monkeypatch):
    monkeypatch.setattr(wf, "planner_agent", lambda task: STEPS)
    monkeypatch.setattr(wf, "developer_agent", lambda steps: ["[WRITE] a.py written"])

    result = run_workflow()

    assert result["status"] == "development_complete"
    assert result["code"] == ["[WRITE] a.py written"]
    assert "error" not in result


# --- /run-task endpoint (workflow replaced by a fake; no LLM calls) -----------
class FakeWorkflow:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def invoke(self, state, config=None):
        self.config = config
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    return TestClient(main.app)


def test_run_task_reports_planner_failure_as_500(client, monkeypatch):
    failed = {"status": "error", "stage": "planner", "error": "LLMError: Groq returned no text"}
    monkeypatch.setattr(main, "setup_graph", lambda: FakeWorkflow(result=failed))

    response = client.post("/run-task", json={"task": "create a file"})

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert body["stage"] == "planner"
    assert body["error"] == "LLMError: Groq returned no text"


def test_run_task_reports_success_with_steps_and_logs(client, monkeypatch):
    done = {"status": "development_complete", "steps": STEPS, "code": ["[WRITE] a.py written"]}
    monkeypatch.setattr(main, "setup_graph", lambda: FakeWorkflow(result=done))

    response = client.post("/run-task", json={"task": "create a file"})

    assert response.status_code == 200
    assert response.json() == {"status": "success", "steps": STEPS, "logs": ["[WRITE] a.py written"]}


def test_run_task_reports_unexpected_exception_as_500(client, monkeypatch):
    monkeypatch.setattr(main, "setup_graph", lambda: FakeWorkflow(error=ValueError("kaput")))

    response = client.post("/run-task", json={"task": "create a file"})

    assert response.status_code == 500
    assert response.json()["stage"] == "workflow"
    assert response.json()["error"] == "ValueError: kaput"


def test_run_task_without_task_returns_400(client, monkeypatch):
    monkeypatch.setattr(main, "setup_graph", lambda: pytest.fail("workflow must not run"))

    response = client.post("/run-task", json={})

    assert response.status_code == 400


# --- planner: optional search-query step (regression for T2b) ------------------
def test_empty_search_query_is_skipped_but_api_errors_still_fail(monkeypatch):
    from agents import planner
    from models.groq_llm import EmptyResponseError, LLMError

    # 1. An empty model answer for the optional search query means "no web search".
    def empty(prompt):
        raise EmptyResponseError("Groq returned no text (model=openai/gpt-oss-20b, finish_reason=stop)")
    monkeypatch.setattr(planner, "chat", empty)
    assert planner.generate_search_query({"enhanced_task": "edit agent_demo.py"}) == {"search_query": ""}

    # 2. With no query, the search decision is skipped without another LLM call.
    def must_not_call(prompt):
        pytest.fail("decide_search_source must not call the LLM when there is no query")
    monkeypatch.setattr(planner, "chat", must_not_call)
    assert planner.decide_search_source({"search_query": "", "task": "t", "codebase_summary": ""}) == {"use_external": False}

    # 3. A genuine LLM failure (API error, rate limit, timeout) is not hidden.
    def api_error(prompt):
        raise LLMError("Groq API error 503: unavailable")
    monkeypatch.setattr(planner, "chat", api_error)
    with pytest.raises(LLMError, match="503"):
        planner.generate_search_query({"enhanced_task": "edit agent_demo.py"})


# --- /run-task: per-task LLM usage and run labels (no real LLM calls) ----------
def test_run_task_logs_per_task_usage_and_names_the_run(client, monkeypatch, caplog):
    from langchain_core.messages import AIMessage
    from models import groq_llm

    class UsingWorkflow(FakeWorkflow):
        def invoke(self, state, config=None):
            self.config = config
            # Simulate two LLM calls made inside the workflow.
            for used in ((100, 20), (40, 10)):
                message = AIMessage(content="x", usage_metadata={
                    "input_tokens": used[0], "output_tokens": used[1], "total_tokens": sum(used)})
                groq_llm._record_usage(message, "prompt", "x", "test-model", "some_node")
            return self.result

    workflow = UsingWorkflow(result={"status": "development_complete", "steps": [], "code": []})
    monkeypatch.setattr(main, "setup_graph", lambda: workflow)
    with caplog.at_level(logging.INFO):
        response = client.post("/run-task", json={"task": "create a file"})

    assert response.status_code == 200
    assert workflow.config == {"run_name": "agentcode_task", "tags": ["run-task"]}
    summary = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Task LLM usage"))
    assert "calls=2 input_tokens=140 output_tokens=30 total_tokens=170" in summary
    assert "groq-reported calls=2, locally estimated calls=0" in summary
