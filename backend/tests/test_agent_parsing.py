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


# --- Whole-file writes must not silently drop existing Python code ----------------
ORIGINAL = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"


def write_step(file, content):
    return {"file": file, "tool": "write", "args": {"content": content}}


def test_write_that_drops_an_existing_function_is_refused_and_file_unchanged(workspace, monkeypatch):
    target = workspace / "groq_modify_test.py"
    target.write_bytes(ORIGINAL.encode("utf-8"))
    bad = write_step("groq_modify_test.py", "def multiply(a, b):\n    return a * b\n\nprint(multiply(2, 3))\n")
    capture_validator(monkeypatch, json.dumps(bad))  # the validator approves the bad step

    logs = developer.run_developer_subgraph([bad])

    assert len(logs) == 1
    assert logs[0].startswith("[ERROR] write on groq_modify_test.py failed")
    assert "would remove existing definitions: add" in logs[0]
    assert target.read_bytes() == ORIGINAL.encode("utf-8")  # byte-for-byte unchanged


def test_additive_write_with_the_complete_file_keeps_existing_code(workspace, monkeypatch):
    (workspace / "groq_modify_test.py").write_text(ORIGINAL, encoding="utf-8")
    complete = ORIGINAL + "\n\ndef multiply(a, b):\n    return a * b\n\nprint(multiply(2, 3))\n"
    step = write_step("groq_modify_test.py", complete)
    capture_validator(monkeypatch, json.dumps(step))

    logs = developer.run_developer_subgraph([step])

    assert logs == ["[WRITE] groq_modify_test.py written"]
    result = (workspace / "groq_modify_test.py").read_text(encoding="utf-8")
    assert result == complete
    assert "def add(a, b):" in result and "print(add(2, 3))" in result
    assert "def multiply(a, b):" in result and "print(multiply(2, 3))" in result


def test_rewrite_that_keeps_every_definition_is_allowed(workspace):
    (workspace / "shapes.py").write_text("class Box:\n    pass\n\ndef area(b):\n    return 0\n", encoding="utf-8")
    rewritten = "def area(b):\n    return b.w * b.h\n\nclass Box:\n    w = 1\n    h = 2\n"

    log = developer.perform_tool_action(developer.ToolStep(**write_step("shapes.py", rewritten)))

    assert log == "[WRITE] shapes.py written"
    assert (workspace / "shapes.py").read_text(encoding="utf-8") == rewritten


@pytest.mark.parametrize("filename,old,new", [
    ("brand_new.py", None, "def only_new():\n    return 1\n"),          # new Python file
    ("notes.txt", "def add(a, b): old notes\n", "completely different\n"),  # non-Python file
    ("page.js", "function add(a, b) { return a + b; }\n", "function multiply(a, b) { return a * b; }\n"),
])
def test_new_and_non_python_files_are_written_as_before(workspace, filename, old, new):
    if old is not None:
        (workspace / filename).write_text(old, encoding="utf-8")

    log = developer.perform_tool_action(developer.ToolStep(**write_step(filename, new)))

    assert log == f"[WRITE] {filename} written"
    assert (workspace / filename).read_text(encoding="utf-8") == new


def test_planner_prompt_says_write_replaces_the_whole_file(monkeypatch):
    seen = []
    monkeypatch.setattr(planner, "chat", lambda prompt: seen.append(prompt) or "[]")

    planner.generate_steps({"codebase_summary": "", "enhanced_task": "add multiply to calc.py"})

    assert '"write" replaces the ENTIRE file' in seen[0]
    assert "the complete updated file" in seen[0]


def test_workflow_reports_developer_error_when_a_write_would_drop_code(workspace, monkeypatch):
    from langgraph_app import graph as main_graph

    target = workspace / "groq_modify_test.py"
    target.write_text(ORIGINAL, encoding="utf-8")
    bad = write_step("groq_modify_test.py", "def multiply(a, b):\n    return a * b\n")
    monkeypatch.setattr(main_graph, "planner_agent", lambda task: [bad])
    capture_validator(monkeypatch, "not json")  # validator keeps the planned step

    result = main_graph.setup_graph().invoke({"task": "add multiply to groq_modify_test.py"})

    assert result["status"] == "error"
    assert result["stage"] == "developer"
    assert "would remove existing definitions: add" in result["error"]
    assert target.read_text(encoding="utf-8") == ORIGINAL

# --- generate_steps() parser reliability ------------------------------------------
# Regression coverage for the merge-sort /run-task failure: a model reply whose
# generated code contains unescaped quotes broke ast.literal_eval and was silently
# swallowed into an empty step list, which graph.py then misreported as
# "Planner produced no steps" even though the model had produced a full plan.
#
# Fixtures build the intended Python VALUE first (with real newlines/quotes in the
# content), then serialize it with json.dumps() / repr() -- the same direction a
# real LLM reply takes -- instead of hand-typing escape sequences in a non-raw
# string literal, which silently turns an intended "\n" escape into an actual
# embedded newline and produces an invalid fixture.


def test_generate_steps_still_parses_existing_python_literal_reply(monkeypatch):
    value = [{"file": "a.py", "tool": "write", "args": {"content": "x\n"}}]
    monkeypatch.setattr(planner, "chat", lambda prompt: repr(value))

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert [s.model_dump() for s in result["steps"]] == value


def test_generate_steps_parses_valid_json_with_multiline_code(monkeypatch):
    value = [{"file": "b.py", "tool": "write", "args": {"content": "def f():\n    return 1\n"}}]
    monkeypatch.setattr(planner, "chat", lambda prompt: json.dumps(value))

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"][0].args["content"] == "def f():\n    return 1\n"


def test_generate_steps_parses_docstring_and_print_with_escaped_quotes(monkeypatch):
    content = 'def f():\n    """doc"""\n    print("hi")\n'
    value = [{"file": "c.py", "tool": "write", "args": {"content": content}}]
    monkeypatch.setattr(planner, "chat", lambda prompt: json.dumps(value))

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"][0].args["content"] == content


def test_generate_steps_preserves_triple_backtick_inside_fenced_content(monkeypatch):
    content = "Example:\n```\ncode here\n```\n"
    value = [{"file": "d.md", "tool": "write", "args": {"content": content}}]
    reply = "```json\n" + json.dumps(value) + "\n```"
    monkeypatch.setattr(planner, "chat", lambda prompt: reply)

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"][0].args["content"] == content


def test_generate_steps_json_fallback_handles_true_false_null(monkeypatch):
    # Not a valid Python literal (bare true/false/null), but valid JSON.
    reply = '[{"file": "e.py", "tool": "write", "args": {"content": "x", "overwrite": true}}]'
    monkeypatch.setattr(planner, "chat", lambda prompt: reply)

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"][0].args == {"content": "x", "overwrite": True}


def _malformed_merge_sort_reply() -> str:
    """Builds a realistic malformed planner reply: newlines are correctly escaped
    (as a real LLM reply typically gets right via JSON encoding), but the
    docstring's own quotes are left unescaped -- the actual mistake that caused
    the reported /run-task merge-sort failure.
    """
    content = 'def merge_sort(arr):\n    """Sort a list."""\n    return arr[:len(arr)//2]\n'
    well_formed = json.dumps([{"file": "shark.py", "tool": "write", "args": {"content": content}}])
    # Un-escape only the embedded quotes (leave the \n newline escapes intact),
    # simulating the model forgetting to escape the docstring's own quotes.
    return well_formed.replace('\\"', '"')


def test_generate_steps_raises_on_unescaped_quote_in_generated_code(monkeypatch):
    # Reproduces the merge-sort failure: an unescaped """ breaks the string boundary
    # and exposes the leaked code's own brackets to the parser.
    monkeypatch.setattr(planner, "chat", lambda prompt: _malformed_merge_sort_reply())

    with pytest.raises(RuntimeError) as excinfo:
        planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert "Could not parse planner output as steps" in str(excinfo.value)


# --- One bounded repair attempt when the planner output fails to parse ----------
# Regression coverage for the real /run-task merge-sort failure: the model left a
# docstring\'s quotes unescaped, generate_steps() correctly raised (proven above),
# but the task still failed outright. These tests cover the recovery path added on
# top of that existing, still-strict parsing: exactly one extra chat() call, with
# the specific parse error and a size-capped snippet of the malformed reply -- never
# the full reply unconditionally -- and the original descriptive RuntimeError is
# still raised, unchanged, if the repair attempt also fails to parse.


def test_generate_steps_repairs_malformed_reply_on_one_retry(monkeypatch):
    calls = []
    valid_value = [{"file": "merge_sort.py", "tool": "write", "args": {"content": "def f(): pass\n"}}]

    def fake_chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return _malformed_merge_sort_reply()
        return json.dumps(valid_value)

    monkeypatch.setattr(planner, "chat", fake_chat)

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert len(calls) == 2
    assert result["steps"][0].file == "merge_sort.py"
    assert result["steps"][0].args["content"] == "def f(): pass\n"
    # The repair call must carry the specific parse error, not a blind retry.
    assert "could not be parsed" in calls[1].lower()
    assert "ast.literal_eval" in calls[1] and "json.loads" in calls[1]


def test_generate_steps_raises_after_repair_attempt_also_fails(monkeypatch):
    calls = []

    def fake_chat(prompt):
        calls.append(prompt)
        return _malformed_merge_sort_reply()

    monkeypatch.setattr(planner, "chat", fake_chat)

    with pytest.raises(RuntimeError) as excinfo:
        planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert len(calls) == 2
    assert "Could not parse planner output as steps" in str(excinfo.value)


def test_repair_prompt_bounds_the_malformed_snippet_size(monkeypatch):
    huge_malformed = "[" + "x" * (planner.REPAIR_SNIPPET_MAX_CHARS * 3)  # invalid and huge
    calls = []

    def fake_chat(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return huge_malformed
        return "[]"

    monkeypatch.setattr(planner, "chat", fake_chat)

    planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert len(calls) == 2
    repair_prompt = calls[1]
    assert len(repair_prompt) < len(huge_malformed)
    assert "truncated" in repair_prompt.lower()


def test_generate_steps_genuinely_empty_list_still_yields_zero_steps(monkeypatch):
    monkeypatch.setattr(planner, "chat", lambda prompt: "[]")

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"] == []


def test_generate_steps_parses_crlf_fenced_reply(monkeypatch):
    value = [{"file": "f.py", "tool": "write", "args": {"content": "y\n"}}]
    reply = "```python\r\n" + repr(value) + "\r\n```"
    monkeypatch.setattr(planner, "chat", lambda prompt: reply)

    result = planner.generate_steps({"codebase_summary": "", "enhanced_task": "t"})

    assert result["steps"][0].file == "f.py"
    assert result["steps"][0].args["content"] == "y\n"


def test_full_planner_subgraph_parse_failure_is_distinguishable_from_no_steps(monkeypatch):
    """Graph-level check: a parsing failure deep in generate_steps() must surface as
    a distinguishable error through run_planner_subgraph -> graph.py's planner_node,
    never as the generic "Planner produced no steps" (reserved for a genuine [] reply).
    """
    from langgraph_app import graph as main_graph

    fake = routed_chat({
        "Enhance this task": "Create shark.py with merge sort.",
        "web search query": "python merge sort",
        "should we fetch external docs": "INTERNAL",
        "software planning agent": _malformed_merge_sort_reply(),
        # The repair attempt also gets the same malformed reply, so this test still
        # exercises the genuine, final-failure path end to end through the graph.
        "could not be parsed as a valid Python list of steps": _malformed_merge_sort_reply(),
    })
    monkeypatch.setattr(planner, "chat", fake)

    result = main_graph.setup_graph().invoke({"task": "create shark.py with merge sort"})

    assert result["status"] == "error"
    assert result["stage"] == "planner"
    assert result["error"] != "Planner produced no steps"
    assert "Could not parse planner output as steps" in result["error"]



# --- validate_step_with_llm() must not rewrite a planned "delete" step ------------
# Regression coverage: "delete" takes no args and nothing in the file's content can
# inform a "correction", so sending it through the LLM validator at all was pure risk
# -- it previously let the model silently turn a planned delete into a write, leaving
# the file on disk while the API still reported success.


def test_validate_step_with_llm_returns_delete_unchanged_without_calling_chat(monkeypatch):
    def fail_if_called(prompt):
        pytest.fail(f"chat() must not be called for a delete step; got prompt: {prompt[:80]!r}")

    monkeypatch.setattr(developer, "chat", fail_if_called)

    original = developer.ToolStep(file="to_delete.txt", tool="delete", args={})
    result = developer.validate_step_with_llm({"current_step": original})

    assert result["current_step"] is original
    assert result["current_step"].file == "to_delete.txt"
    assert result["current_step"].tool == "delete"
    assert result["current_step"].args == {}


def test_delete_step_executes_without_validator_rewriting_it(workspace, monkeypatch):
    (workspace / "to_delete.txt").write_text("x\n", encoding="utf-8")

    def fail_if_called(prompt):
        pytest.fail("chat() must not be called for a delete step")

    monkeypatch.setattr(developer, "chat", fail_if_called)

    logs = developer.run_developer_subgraph([
        {"file": "to_delete.txt", "tool": "delete", "args": {}}
    ])

    assert logs == ["[DELETE] to_delete.txt removed"]
    assert not (workspace / "to_delete.txt").exists()
