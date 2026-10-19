# Agent Code: an agentic code editor

Agent Code is a browser-based code editor with an AI agent that plans and applies file changes (create, modify, delete) in a local workspace, and a Run button that executes code in a self-hosted [Piston](https://github.com/engineer-man/piston) sandbox.

> Demo video (may not reflect the current version): [Google Drive](https://drive.google.com/drive/folders/1bqPaS2Cie13lI22dzSma5gW-o7jmXS8H?usp=sharing)

---

## Features

- **Chat panel**: messages go to a Groq-hosted chat model. If the model classifies a message as a code-change request, the chat shows an **Execute Work** button that sends the request to the agent.
- **File agent**: a LangGraph pipeline (planner → developer) turns the request into file steps and applies them inside `codebase/`.
- **Code editor and file explorer**: Monaco editor with one tab per open file; the explorer can create files and folders, rename and delete them; the editor saves with the Save button or Ctrl+S. All paths stay inside `codebase/`.
- **One-click Run**: runs the open file through the backend and shows its output, for Python, JavaScript, TypeScript, Java, C#, C and C++ (versions below).
- **Safety rails**:
  - every file operation (UI and agent) is confined to `codebase/`;
  - agent-written `.py` files that do not parse are refused before saving;
  - workflow and model failures are returned as errors instead of a false success.

See [Verification](#verification) for what has been tested and how.

---

## Architecture

```
Browser (React + Vite, Chakra UI, Monaco)
  │
  ├─ chat ──────────► POST /chat       → chat model (Groq, openai/gpt-oss-120b via langchain-groq)
  │
  ├─ Execute Work ──► POST /run-task   → LangGraph workflow
  │                                        planner   (Gemini 2.5 Flash, google-genai SDK; optional Tavily web search)
  │                                        developer (Gemini 2.5 Flash; checks each step, then runs it)
  │                                        → file operations confined to codebase/
  │
  ├─ explorer ──────► /fs/*            → list, read, create, rename, delete, save inside codebase/
  │
  └─ Run ───────────► POST /execute    → allow-listed runtimes → self-hosted Piston (http://127.0.0.1:2000)
```

The backend is FastAPI. It never executes submitted code itself: `/execute` checks the language and version against an allow-list (400 if not allowed), forwards the request to Piston, and maps failures to clear errors: 502 if Piston rejects the request, 503 if Piston is unreachable, 504 if it times out.

---

## Setup

### Prerequisites
- Python 3.12 (development and tests used 3.12.4)
- Node.js 18 or newer, with npm (the frontend uses Vite 5)
- Docker, to run Piston locally
- API keys: Google Gemini and Groq (Tavily is optional)

### 1. Self-hosted Piston (required for Run)
The public Piston API is not used. Run your own Piston container by following the [Piston README](https://github.com/engineer-man/piston), so that its API is reachable at `http://127.0.0.1:2000`. Then install these runtimes with Piston's package manager. The backend accepts only these exact versions:

| Language (as sent by the UI) | Version | Piston package |
|---|---|---|
| `python` | 3.12.0 | python |
| `javascript` | 18.15.0 | node |
| `typescript` | 5.0.3 | typescript |
| `java` | 15.0.2 | java |
| `csharp` | 6.12.0 | mono |
| `c`, `cpp` | 10.2.0 | gcc |

C++ compilation can exceed Piston's default compile timeout. If it does, raise the limit with Piston's `PISTON_LIMIT_OVERRIDES` environment variable. Example value: `PISTON_LIMIT_OVERRIDES={"c++":{"compile_timeout":30000}}` (30 s compile timeout for C++).

### 2. Backend
Open a terminal in the repository root (the folder that contains `backend/` and `frontend/`), then:
```bash
cd backend
python -m venv .venv
```
Activate the virtual environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Still in `backend/`, install the dependencies and create your `.env` file (`cp` works in PowerShell and bash; in Windows cmd use `copy .env.example .env`):
```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `backend/.env`:

| Variable | Used by |
|---|---|
| `GOOGLE_GENAI_API_KEY` | Planner and developer agents (Gemini 2.5 Flash). Required for agent tasks. |
| `GROQ_API_KEY` | Chat panel and code-change detection. Required for chat. |
| `TAVILY_API_KEY` | Optional web search in the planner. Without it, the search returns no results and planning continues. |

In the same terminal (still in `backend/`, with the virtual environment active), start the API:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Frontend
In a second terminal, starting from the repository root:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. The backend's CORS setting allows only this origin, and the frontend expects the backend at port 8000.

### Running the tests
In a terminal in `backend/`, with the virtual environment active:
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Verification

**Automated tests:** `50 passed, 1 skipped` on the merged `main`. The skipped test needs permission to create symlinks on Windows. The tests cover:
- workspace confinement for the UI endpoints and the agent's file helpers;
- the `/execute` proxy and its error mapping;
- Gemini errors, timeouts and empty responses, with a bounded retry;
- skipping the optional web search when the model returns no query;
- workflow error reporting;
- refusal of an agent edit that would leave broken Python.

The model calls are replaced with test doubles, so the tests make no API calls.

**Manual end-to-end runs:** on Windows, with the real models and a local Piston container. Each run was a single attempt, with file hashes checked before and after:

| Check | Result |
|---|---|
| Agent creates a file (via the API) | PASS |
| Agent modifies a file (via the API) | PASS after two fixes; the earlier attempts failed (an edit that left broken Python, then an empty model response) |
| Agent deletes a file (via the API) | PASS |
| UI: chat → Execute Work → new file appears in the explorer → Run | PASS |
| UI Run button, one hello-world per language (7 languages) | 7/7 PASS: expected output, and HTTP 200 from `/execute` |

These are single runs against a non-deterministic model: they show the paths work, not that every request will succeed. A first create run failed before the honest-reporting fix: the failure was hidden and success reported. The detailed test record, including the failed runs and evidence files, is kept in the author's project notes and is not part of this repository.

---

## Known limitations

- **Agent results depend on the model.** Plans and edits come from Gemini, and a request can still be planned or applied incorrectly. There is no plan review before the agent edits files, and no automatic check that the result matches the request.
- **The UI sends the task as an object** (`{"text": ..., "id": ...}`) instead of plain text. The planner handled it in testing, but the payload has not been cleaned up.
- **No stdin.** The Run panel shows only the output text, not the exit code or signal.
- **No AI autocomplete.** The editor offers Monaco's built-in, non-AI suggestions only.
- **Execution is request/response**, not streamed: output appears when the run finishes.
- **Long plans can fail.** No recursion limit is configured, so LangGraph's default limits how many developer steps a single task can run.
- **Not yet verified end to end:**
  - modifying and deleting files through the UI (verified through the API only);
  - the explorer's folder, rename and delete actions (endpoints covered by unit tests only);
  - a live test of workspace confinement (unit tests only).
- **Local development setup only:**
  - the backend URLs (`localhost:8000` / `127.0.0.1:8000`), the Piston URL and the CORS origin are hard-coded;
  - blocking calls run inside async endpoints, so a long agent task holds up other requests.

---

## Tech stack

- **Frontend:** React, Vite, Chakra UI, Monaco Editor
- **Backend:** FastAPI, httpx, Pydantic
- **Agents:** LangGraph (planner → developer); Gemini 2.5 Flash via the `google-genai` SDK; Groq `openai/gpt-oss-120b` via `langchain-groq` for chat
- **Code execution:** self-hosted Piston
- **Tests:** pytest
