import os
import ast
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter()


class PathRequest(BaseModel):
    path: str


# --- Workspace confinement ---------------------------------------------------
# Every file operation (UI endpoints AND agent helpers) must resolve its path
# through resolve_in_workspace(), so there is a single place where the
# "stay inside codebase/" rule is enforced.
WORKSPACE = (Path(__file__).resolve().parents[2] / "codebase").resolve()


class PathOutsideWorkspace(ValueError):
    """Raised when a requested path resolves outside the workspace."""


def resolve_in_workspace(rel_path: str) -> Path:
    # resolve() collapses ".." and follows symlinks; an absolute rel_path
    # replaces WORKSPACE entirely and is then rejected by the check below.
    candidate = (WORKSPACE / rel_path).resolve()
    # Compare whole path components (not string prefixes), so a sibling
    # folder such as "codebase_secrets" is not treated as inside "codebase".
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise PathOutsideWorkspace(f"Path is outside the workspace: {rel_path}")
    return candidate


def _http_error(exc: Exception) -> HTTPException:
    # Keep deliberate HTTP errors (e.g. 404) as they are, map confinement
    # violations to 403, and keep the previous 400 for everything else.
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PathOutsideWorkspace):
        return HTTPException(status_code=403, detail="Access denied")
    return HTTPException(status_code=400, detail=str(exc))


# --- UI endpoints -------------------------------------------------------------
@router.get("/fs/list")
def list_dir(path: str = ""):
    try:
        abs_path = resolve_in_workspace(path)
        items = []
        for name in os.listdir(abs_path):
            full_path = os.path.join(abs_path, name)
            items.append({
                "name": name,
                "type": "folder" if os.path.isdir(full_path) else "file"
            })
        return items
    except Exception as e:
        raise _http_error(e)

@router.post("/fs/create_folder")
def create_folder(req: PathRequest):
    try:
        abs_path = resolve_in_workspace(req.path)
        os.makedirs(abs_path, exist_ok=True)
        return {"success": True}
    except Exception as e:
        raise _http_error(e)

@router.post("/fs/create_file")
def create_file(req: PathRequest):
    try:
        abs_path = resolve_in_workspace(req.path)
        # "x" = exclusive create: fails instead of emptying an existing file.
        open(abs_path, "x").close()
        return {"success": True}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="File already exists")
    except Exception as e:
        raise _http_error(e)

@router.post("/fs/delete")
def delete_path(req: PathRequest):
    try:
        abs_path = resolve_in_workspace(req.path)
        if abs_path == WORKSPACE:
            raise HTTPException(status_code=400, detail="Cannot delete the workspace root")
        if os.path.isdir(abs_path):
            shutil.rmtree(abs_path)
        else:
            os.remove(abs_path)
        return {"success": True}
    except Exception as e:
        raise _http_error(e)


# --- Agent helpers (used by agents/planner.py and agents/developer.py) -------
def list_code_files():
    return [f for f in os.listdir(WORKSPACE) if os.path.isfile(os.path.join(WORKSPACE, f))]

def read_code_file(filename: str) -> list[str]:
    path = resolve_in_workspace(filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_code_file(filename: str, lines: list[str]):
    path = resolve_in_workspace(filename)
    if path.suffix == ".py":
        ast.parse("".join(lines), filename=filename)  # refuse to save invalid Python
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def apply_change(filename: str, action: str, line: int, new_code: str = None):
    lines = read_code_file(filename)
    if action == "modify":
        lines[line-1] = new_code + "\n"
    elif action == "insert":
        lines.insert(line, new_code + "\n")
    elif action == "delete":
        lines.pop(line-1)
    write_code_file(filename, lines)

def code_to_ast_string(code: str) -> str:
    try:
        tree = ast.parse(code)
        return ast.dump(tree, indent=4)
    except Exception as e:
        return f"Could not parse AST: {e}"

def delete_code_file(filename: str):
    path = resolve_in_workspace(filename)
    if os.path.exists(path):
        os.remove(path)


class RenameRequest(BaseModel):
    old_path: str
    new_path: str

@router.post("/fs/rename")
def rename_path(req: RenameRequest):
    try:
        abs_old = resolve_in_workspace(req.old_path)
        abs_new = resolve_in_workspace(req.new_path)
        if WORKSPACE in (abs_old, abs_new):
            raise HTTPException(status_code=400, detail="Cannot rename the workspace root")
        os.rename(abs_old, abs_new)
        return {"success": True}
    except Exception as e:
        raise _http_error(e)

@router.get("/fs/read")
def read_file(path: str):
    try:
        abs_path = resolve_in_workspace(path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="File not found")
        with open(abs_path, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read())
    except Exception as e:
        raise _http_error(e)

class SaveFileRequest(BaseModel):
    path: str
    content: str

@router.post("/fs/save")
def save_file(req: SaveFileRequest):
    try:
        abs_path = resolve_in_workspace(req.path)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(req.content)
        return {"success": True}
    except Exception as e:
        raise _http_error(e)
