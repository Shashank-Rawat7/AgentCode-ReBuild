"""Planner/developer output parsing with the LLM replaced by a local fake.

These tests check that typical Groq-style replies (fenced code blocks, Python
lists, JSON objects) still flow through the existing parsers and tool
dispatcher. No network call is made and no API key is needed.
"""

import pytest

from agents import developer, planner
from utils import file_ops


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = (tmp_path / "codebase")
    ws.mkdir()
    ws = ws.resolve()
    monkeypatch.setattr(file_ops, "WORKSPACE", ws)
    return ws


def routed_chat(replies):
    """Return a fake chat() that answers by matching a phrase in each prompt."""
    seen = []

    def fake_chat(prompt):
        for phrase, answer in replies.items():
            if phrase in prompt:
                seen.append(phrase)
                return answer
        pytest.fail(f"unexpected prompt: {prompt[:80]!r}")

    fake_chat.seen = seen
    return fake_chat


def test_planner_parses_fenced_python_list_into_tool_steps(workspace, monkeypatch):
    (workspace / "agent_demo.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    plan = (
        "```python\n"
        "[{'file': 'agent_demo.py', 'tool': 'write', "
        "'args': {'content': 'def add(a, b):\\n    return a + b\\n\\nprint(add(2, 3))\\n'}}]\n"
        "```"
    )
    fake = routed_chat({
        "Enhance this task": "Add a final print(add(2, 3)) line to agent_demo.py.",
        "web search query": "python print function",
        "should we fetch external docs": "INTERNAL",
        "software planning agent": plan,
    })
    monkeypatch.setattr(planner, "chat", fake)

    steps = planner.run_planner_subgraph("add a print line to agent_demo.py")

    assert steps == [{
        "file": "agent_demo.py",
        "tool": "write",
        "args": {"content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"},
    }]
    assert fake.seen == ["Enhance this task", "web search query", "should we fetch external docs", "software planning agent"]


def test_developer_applies_fenced_json_step_through_tool_dispatcher(workspace, monkeypatch):
    corrected = (
        "```json\n"
        '{"file": "demo.py", "tool": "write", "args": {"content": "print(\\"hi\\")\\n"}}\n'
        "```"
    )
    monkeypatch.setattr(developer, "chat", routed_chat({"code step validator": corrected}))

    logs = developer.run_developer_subgraph([{"file": "demo.py", "tool": "llm_modify", "args": {}}])

    assert logs == ["[WRITE] demo.py written"]
    assert (workspace / "demo.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_developer_keeps_original_step_when_reply_is_not_json(workspace, monkeypatch):
    (workspace / "keep.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(developer, "chat", routed_chat({"code step validator": "Sure! Here is the step."}))

    logs = developer.run_developer_subgraph([{"file": "keep.py", "tool": "read", "args": {}}])

    assert logs == ["[READ] keep.py has 1 lines"]
    assert (workspace / "keep.py").read_text(encoding="utf-8") == "x = 1\n"


def test_enhance_and_search_decision_prompts_exclude_the_codebase(workspace, monkeypatch):
    # A large, unique marker stands in for the source files: it must reach generate_steps only.
    marker = "CODEBASE_MARKER_" + "x" * 2000
    (workspace / "big.py").write_text(f"# {marker}\n", encoding="utf-8")
    prompts = {}

    def fake_chat(prompt):
        for phrase, answer in {
            "Enhance this task": "Create hello.py that prints hello.",
            "web search query": "python print",
            "should we fetch external docs": "INTERNAL",
            "software planning agent": "[]",
        }.items():
            if phrase in prompt:
                prompts[phrase] = prompt
                return answer
        pytest.fail(f"unexpected prompt: {prompt[:80]!r}")

    monkeypatch.setattr(planner, "chat", fake_chat)
    planner.run_planner_subgraph("create hello.py that prints hello")

    assert marker not in prompts["Enhance this task"]
    assert "create hello.py that prints hello" in prompts["Enhance this task"]
    assert marker not in prompts["should we fetch external docs"]
    assert "Task: create hello.py that prints hello" in prompts["should we fetch external docs"]
    assert "Enhanced task: Create hello.py that prints hello." in prompts["should we fetch external docs"]
    assert marker not in prompts["web search query"]
    assert marker in prompts["software planning agent"]  # planning still sees the codebase


@pytest.mark.parametrize("answer,expected", [("EXTERNAL", True), (" internal ", False), ("external\n", True)])
def test_search_decision_still_maps_external_and_internal(monkeypatch, answer, expected):
    seen = []
    monkeypatch.setattr(planner, "chat", lambda prompt: seen.append(prompt) or answer)
    state = {"task": "t", "enhanced_task": "e", "search_query": "q", "codebase_summary": "SHOULD_NOT_BE_SENT"}
    assert planner.decide_search_source(state) == {"use_external": expected}
    assert "SHOULD_NOT_BE_SENT" not in seen[0]
