import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from utils import file_ops
from utils.file_ops import PathOutsideWorkspace, resolve_in_workspace


# Every test runs against a throwaway workspace under pytest's tmp_path.
# The real codebase/ folder is never touched.
@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "codebase"
    ws.mkdir()
    ws = ws.resolve()
    monkeypatch.setattr(file_ops, "WORKSPACE", ws)
    return ws


@pytest.fixture
def client(workspace):
    # Mount only the file-ops router, so no LLM clients are created.
    app = FastAPI()
    app.include_router(file_ops.router)
    return TestClient(app)


# --- resolve_in_workspace -----------------------------------------------------
def test_resolve_allows_paths_inside_workspace(workspace):
    assert resolve_in_workspace("src/app.py") == workspace / "src" / "app.py"
    assert resolve_in_workspace(".") == workspace
    assert resolve_in_workspace("./a.py") == workspace / "a.py"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "a/../../outside.txt",
        "../codebase_secrets/x.txt",  # sibling folder sharing the "codebase" prefix (A2)
    ],
)
def test_resolve_rejects_paths_that_escape(workspace, path):
    (workspace.parent / "codebase_secrets").mkdir(exist_ok=True)
    with pytest.raises(PathOutsideWorkspace):
        resolve_in_workspace(path)


def test_resolve_rejects_absolute_path(workspace, tmp_path):
    with pytest.raises(PathOutsideWorkspace):
        resolve_in_workspace(str(tmp_path / "outside.txt"))


def test_resolve_rejects_symlink_that_points_outside(workspace, tmp_path):
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    try:
        os.symlink(outside, workspace / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this machine")
    with pytest.raises(PathOutsideWorkspace):
        resolve_in_workspace("link/secret.txt")


# --- agent helpers (A1) ---------------------------------------------------------
def test_agent_read_write_inside_workspace_still_work(workspace):
    file_ops.write_code_file("hello.py", ["print('hi')\n"])
    assert file_ops.read_code_file("hello.py") == ["print('hi')\n"]
    assert file_ops.list_code_files() == ["hello.py"]


def test_agent_write_outside_workspace_is_blocked(workspace, tmp_path):
    with pytest.raises(PathOutsideWorkspace):
        file_ops.write_code_file("../escape.txt", ["x"])
    assert not (tmp_path / "escape.txt").exists()


def test_agent_delete_outside_workspace_is_blocked(workspace, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    with pytest.raises(PathOutsideWorkspace):
        file_ops.delete_code_file("../victim.txt")
    assert victim.read_text() == "keep me"


@pytest.mark.parametrize(
    "tool,args",
    [
        ("write", {"content": "overwritten"}),
        ("delete", {}),
        ("apply_change", {"action": "modify", "line": 1, "new_code": "overwritten"}),
    ],
)
def test_developer_step_with_escaping_path_is_logged_as_error(workspace, tmp_path, tool, args):
    # Runs the developer's tool step directly (run_code_update ->
    # perform_tool_action -> file_ops helper); the LLM validator node is
    # never invoked, so no LLM call is made.
    from agents.developer import ToolStep, run_code_update

    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    before = sorted(p.name for p in tmp_path.iterdir())

    step = ToolStep(file="../victim.txt", tool=tool, args=args)
    result = run_code_update({"current_step": step, "logs": []})

    log = result["logs"][0]
    assert log.startswith("[ERROR]")
    assert "outside the workspace" in log  # the confinement check fired
    assert victim.read_text() == "keep me"  # not overwritten or deleted
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing created outside
    assert list(workspace.iterdir()) == []  # workspace untouched


# --- UI endpoints ---------------------------------------------------------------
def test_api_normal_file_operations_still_work(client, workspace):
    assert client.post("/fs/create_folder", json={"path": "src"}).status_code == 200
    assert client.post("/fs/create_file", json={"path": "src/a.py"}).status_code == 200
    assert client.post("/fs/save", json={"path": "src/a.py", "content": "x = 1\n"}).status_code == 200
    assert client.get("/fs/read", params={"path": "src/a.py"}).text == "x = 1\n"
    assert client.post("/fs/rename", json={"old_path": "src/a.py", "new_path": "src/b.py"}).status_code == 200
    assert {"name": "b.py", "type": "file"} in client.get("/fs/list", params={"path": "src"}).json()
    assert client.post("/fs/delete", json={"path": "src/b.py"}).status_code == 200
    assert not (workspace / "src" / "b.py").exists()


def test_api_read_outside_workspace_returns_403(client):
    assert client.get("/fs/read", params={"path": "../outside.txt"}).status_code == 403


def test_api_rename_target_outside_workspace_returns_403(client, workspace, tmp_path):
    (workspace / "a.py").write_text("x")
    response = client.post("/fs/rename", json={"old_path": "a.py", "new_path": "../a.py"})
    assert response.status_code == 403
    assert (workspace / "a.py").exists()
    assert not (tmp_path / "a.py").exists()


def test_api_read_missing_file_returns_404(client):
    assert client.get("/fs/read", params={"path": "missing.py"}).status_code == 404


def test_api_delete_workspace_root_is_refused(client, workspace):
    (workspace / "keep.py").write_text("x")
    response = client.post("/fs/delete", json={"path": "."})
    assert response.status_code == 400
    assert (workspace / "keep.py").exists()


def test_api_rename_workspace_root_is_refused(client, workspace):
    response = client.post("/fs/rename", json={"old_path": ".", "new_path": "moved"})
    assert response.status_code == 400
    assert workspace.is_dir()


def test_api_create_existing_file_returns_409_and_keeps_content(client, workspace):
    (workspace / "a.py").write_text("important")
    response = client.post("/fs/create_file", json={"path": "a.py"})
    assert response.status_code == 409
    assert (workspace / "a.py").read_text() == "important"


@pytest.mark.parametrize(
    "url,body",
    [
        ("/fs/delete", {"path": "../victim.txt"}),
        ("/fs/save", {"path": "../victim.txt", "content": "overwritten"}),
        ("/fs/create_file", {"path": "../new.txt"}),
        ("/fs/create_folder", {"path": "../newdir"}),
        ("/fs/rename", {"old_path": "../victim.txt", "new_path": "moved.txt"}),
        ("/fs/rename", {"old_path": "keep.py", "new_path": "../moved.txt"}),
    ],
)
def test_api_mutations_outside_workspace_return_403_and_change_nothing(
    client, workspace, tmp_path, url, body
):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (workspace / "keep.py").write_text("inside")
    outside_before = sorted(p.name for p in tmp_path.iterdir())

    response = client.post(url, json=body)

    assert response.status_code == 403
    assert victim.read_text() == "keep me"  # not deleted, overwritten or moved
    assert sorted(p.name for p in tmp_path.iterdir()) == outside_before  # nothing created outside
    assert [p.name for p in workspace.iterdir()] == ["keep.py"]  # workspace unchanged
    assert (workspace / "keep.py").read_text() == "inside"


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_resolve_rejects_windows_junction_that_points_outside(workspace, tmp_path):
    import subprocess

    outside = tmp_path / "outside_dir"
    outside.mkdir()
    junction = workspace / "junction"
    # "mklink /J" creates a directory junction; it does not need admin rights.
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not junction.exists():
        pytest.skip(f"could not create a junction: {completed.stderr.strip()}")

    with pytest.raises(PathOutsideWorkspace):
        resolve_in_workspace("junction/secret.txt")


# --- Agent edits must not save broken Python (regression for T2) ---------------
def test_agent_edit_that_breaks_python_is_rejected_and_file_unchanged(workspace):
    # Replays the exact T2 step: inserting a bare "def multiply(a, b):" after the
    # last line leaves a function with no body (IndentationError).
    from agents.developer import ToolStep, run_code_update

    demo = workspace / "agent_demo.py"
    demo.write_text("def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n", encoding="utf-8")
    before = demo.read_bytes()

    step = ToolStep(file="agent_demo.py", tool="apply_change",
                    args={"action": "insert", "line": 4, "new_code": "def multiply(a, b):"})
    result = run_code_update({"current_step": step, "logs": []})

    assert result["logs"][-1].startswith("[ERROR] apply_change on agent_demo.py failed")
    assert demo.read_bytes() == before  # the original file was not overwritten
