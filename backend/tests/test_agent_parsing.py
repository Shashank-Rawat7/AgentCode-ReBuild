"""Planner/developer output parsing with the LLM replaced by a local fake.

These tests check that typical Groq-style replies (fenced code blocks, Python
lists, JSON objects) still flow through the existing parsers and tool
dispatcher. No network call is made and no API key is needed.
"""

import json

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


# --- Developer validation prompt: target file only ---------------------------------
def capture_validator(monkeypatch, answer):
    prompts = []

    def fake_chat(prompt):
        assert "code step validator" in prompt
        prompts.append(prompt)
        return answer

    monkeypatch.setattr(developer, "chat", fake_chat)
    return prompts


def test_validator_prompt_has_target_file_but_not_other_file_contents(workspace, monkeypatch):
    (workspace / "target.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    marker = "UNRELATED_MARKER_" + "y" * 2000
    (workspace / "other.py").write_text(f"# {marker}\n", encoding="utf-8")
    new_content = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"
    corrected = json.dumps({"file": "target.py", "tool": "write", "args": {"content": new_content}})
    prompts = capture_validator(monkeypatch, corrected)

    logs = developer.run_developer_subgraph([{"file": "target.py", "tool": "llm_modify", "args": {}}])

    prompt = prompts[0]
    assert "Current contents of target.py:\ndef add(a, b):\n    return a + b\n" in prompt
    assert "other.py" in prompt          # listed by name
    assert marker not in prompt          # but its contents are not sent
    assert "'tool': 'llm_modify'" in prompt  # the planned step is still included
    assert logs == ["[WRITE] target.py written"]
    assert (workspace / "target.py").read_text(encoding="utf-8") == new_content
    assert (workspace / "other.py").read_text(encoding="utf-8") == f"# {marker}\n"


def test_validator_prompt_marks_a_new_file_and_the_write_still_works(workspace, monkeypatch):
    corrected = json.dumps({"file": "new_file.py", "tool": "write", "args": {"content": "print('new')\n"}})
    prompts = capture_validator(monkeypatch, corrected)

    logs = developer.run_developer_subgraph([{"file": "new_file.py", "tool": "write", "args": {"content": "print('new')\n"}}])

    assert "Current contents of new_file.py:\n(file does not exist yet)" in prompts[0]
    assert logs == ["[WRITE] new_file.py written"]
    assert (workspace / "new_file.py").read_text(encoding="utf-8") == "print('new')\n"


def test_validator_does_not_read_files_outside_the_workspace(workspace, monkeypatch):
    secret = workspace.parent / "outside_secret.txt"
    secret.write_text("OUTSIDE_SECRET_CONTENT\n", encoding="utf-8")
    prompts = capture_validator(monkeypatch, "not json")  # keep the original step

    logs = developer.run_developer_subgraph([{"file": "../outside_secret.txt", "tool": "read", "args": {}}])

    assert "OUTSIDE_SECRET_CONTENT" not in prompts[0]
    assert "Current contents of ../outside_secret.txt:\n(could not be read)" in prompts[0]
    assert logs[0].startswith("[ERROR] read on ../outside_secret.txt failed")  # still refused by file_ops
