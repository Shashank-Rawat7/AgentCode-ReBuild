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
  │                                        planner   (Groq model via langchain-groq; optional Tavily web search)
  │                                        developer (Groq model; checks each step, then runs it)
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
- A Groq API key (Tavily is optional)

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
| `GROQ_API_KEY` | Planner and developer agents, chat panel and code-change detection. Required. |
| `GROQ_MODEL` | Model used by the planner and developer agents. Optional; defaults to `openai/gpt-oss-20b`. The chat panel keeps its own model (`openai/gpt-oss-120b`). |
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

### Optional: token usage and LangSmith tracing
- **Token usage** is always logged by the backend. Each LLM call writes a line like `LLM usage purpose=generate_steps model=... input_tokens=... output_tokens=... total_tokens=... source=groq-reported`, and each `/run-task` request ends with a `Task LLM usage: calls=... total_tokens=...` summary. `source=groq-reported` means the numbers came from Groq; `source=local-estimate` (about 4 characters per token) is used only if Groq reports no usage.
- **Rate limits (429):** the agent retries once only when Groq's suggested wait is short (`AGENTCODE_MAX_429_WAIT_SECONDS`, default 20); otherwise it fails at once with Groq's message.
- **Prompt-size guard:** `AGENTCODE_MAX_PROMPT_TOKENS` (default `0`, disabled). When set, prompts whose local estimate exceeds it are not sent.
- **LangSmith tracing** is off by default. To see the graph, each node's inputs and outputs, and each LLM call in [LangSmith](https://smith.langchain.com), set in `backend/.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=<your key>` and optionally `LANGSMITH_PROJECT=agentcode`, then restart the backend. Each task appears as a run named `agentcode_task`. Traces contain prompts and file contents, so only enable tracing for code you are happy to send to LangSmith.

---

## Verification

**Automated tests:** `78 passed, 2 skipped`. One skipped test needs permission to create symlinks on Windows; the other is the optional live Groq test (below). The tests cover:
- workspace confinement for the UI endpoints and the agent's file helpers;
- the `/execute` proxy and its error mapping;
- Groq API errors, timeouts and empty responses, with a bounded retry;
- token-usage logging (Groq-reported vs estimated), per-task totals, the 429 retry rule and the optional prompt-size guard;
- skipping the optional web search when the model returns no query;
- workflow error reporting;
- refusal of an agent edit that would leave broken Python;
- planner and developer parsing of typical model replies (fenced Python lists and JSON), through to the tool dispatcher.

The model calls are replaced with test doubles, so these tests make no API calls. An optional live test makes one real Groq call; it is skipped unless enabled (from `backend/`, PowerShell: `$env:AGENTCODE_LIVE_GROQ = "1"; python -m pytest -q tests/test_live_groq.py`).

**Manual end-to-end runs:** on Windows, with a local Piston container. These runs used the earlier Gemini-based agent, before the switch to Groq, and have not been repeated with Groq, apart from the single Groq run below. Each run was a single attempt, with file hashes checked before and after:

| Check | Result |
|---|---|
| Agent creates a file (via the API) | PASS |
| Agent modifies a file (via the API) | PASS after two fixes; the earlier attempts failed (an edit that left broken Python, then an empty model response) |
| Agent deletes a file (via the API) | PASS |
| UI: chat → Execute Work → new file appears in the explorer → Run | PASS |
| UI Run button, one hello-world per language (7 languages) | 7/7 PASS: expected output, and HTTP 200 from `/execute` |

These are single runs against a non-deterministic model: they show the paths work, not that every request will succeed. A first create run failed before the honest-reporting fix: the failure was hidden and success reported. The detailed test record, including the failed runs and evidence files, is kept in the author's project notes and is not part of this repository.

**Groq live run (token-usage logging):** one UI task through the Groq-based agent, `Create a new file named usage_check.py containing print("usage check")`, succeeded and the file was written. The backend logged five LLM calls, all with `source=groq-reported`, and the per-call totals add up to the task summary:

| Node | Total tokens |
|---|---|
| `enhance_prompt` | 307 |
| `generate_search_query` | 495 |
| `decide_search_source` | 200 |
| `generate_steps` | 3,298 |
| `validate_step_with_llm` | 3,393 |
| **Task total** | **7,693** (6,863 input, 830 output) |

This is a single run. It shows that usage logging works; it does not show any token reduction or a fix for 429 errors. The two codebase-aware calls (`generate_steps` and `validate_step_with_llm`) used most of the tokens, and the task total is close to the 8,000 tokens-per-minute limit we hit on Groq's free tier.

---

## Known limitations

- **Agent results depend on the model.** Plans and edits come from the Groq-hosted model, and a request can still be planned or applied incorrectly. There is no plan review before the agent edits files, and no automatic check that the result matches the request.
- **The UI sends the task as an object** (`{"text": ..., "id": ...}`) instead of plain text. The planner handled it in testing, but the payload has not been cleaned up.
- **No stdin.** The Run panel shows only the output text, not the exit code or signal.
- **No AI autocomplete.** The editor offers Monaco's built-in, non-AI suggestions only.
- **Execution is request/response**, not streamed: output appears when the run finishes.
- **Groq rate limits apply.** On the free tier, a task can hit HTTP 429 (tokens per minute). The agent retries once only when Groq suggests a short wait; otherwise it reports the error. The planner and developer still send full file contents, so larger tasks are the most likely to hit the limit.
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
- **Agents:** LangGraph (planner → developer); Groq via `langchain-groq`: `openai/gpt-oss-20b` by default for the agents (configurable with `GROQ_MODEL`), `openai/gpt-oss-120b` for chat
- **Code execution:** self-hosted Piston
- **Tests:** pytest
