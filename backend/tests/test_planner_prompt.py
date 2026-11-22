"""Focused regression test for the Milestone-3 /run-task smoke-test failure:
"Planner produced no steps" when the task asked for a file inside an
already-existing subdirectory. The fix is prompt-only (generate_steps() in
agents/planner.py); this test checks the prompt text itself, not the live
model, so it needs no Groq calls and no network access.
"""

from agents import planner


def test_generate_steps_prompt_covers_file_creation_and_subdirectories(monkeypatch):
    captured = {}

    def fake_chat(prompt):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(planner, "chat", fake_chat)

    planner.generate_steps({
        "codebase_summary": "",
        "enhanced_task": "Create a new file named notes.txt inside the existing folder docker_agent_smoke_test",
        "external_results": [],
    })

    prompt = captured["prompt"]

    # 1. Creating a file is explicitly actionable -- must become a "write" step.
    assert "actionable step" in prompt
    assert 'must produce a "write" step' in prompt

    # 2. "file" may include a subdirectory path, including one that already exists.
    assert "relative subdirectory path" in prompt
    assert "already exists in the codebase" in prompt

    # 3. A simple file-creation request gets exactly one "write" step, with a
    #    concrete example using the real ToolStep schema (file/tool/args).
    assert '[{"file": "docker_agent_smoke_test/notes.txt", "tool": "write", "args": {"content": "hello\\n"}}]' in prompt

    # 4. [] is reserved for genuinely non-actionable tasks, not missing tools.
    assert "genuinely no actionable file operation" in prompt
    assert 'no explicit "create directory" tool exists' in prompt
