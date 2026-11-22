# Agent Code: an agentic code editor

Agent Code is a browser-based code editor with an AI agent that plans and applies file changes (create, modify, delete) in a local workspace, and a Run button that executes code in a self-hosted [Piston](https://github.com/engineer-man/piston) sandbox.

> Demo video (may not reflect the current version): [Google Drive](https://drive.google.com/drive/folders/1bqPaS2Cie13lI22dzSma5gW-o7jmXS8H?usp=sharing)

---

## Features

- **Chat panel**: messages go to a Groq-hosted chat model. If the model classifies a message as a code-change request, the chat shows an **Execute Work** button that sends the request to the agent.
- **File agent**: a LangGraph pipeline (planner → developer) turns the request into file steps and applies them inside `codebase/`, including nested subfolders.
- **Code editor and file explorer**: Monaco editor with one tab per open file; the explorer can create files and folders, rename and delete them; the editor saves with the Save button or Ctrl+S. All paths stay inside `codebase/`.
- **One-click Run**: runs the open file through the backend and shows its output, for Python, JavaScript, TypeScript, Java, C#, C and C++ (versions below).
- **Safety rails**:
  - every file operation (UI and agent) is confined to `codebase/`;
  - agent-written `.py` files that do not parse are refused before saving;
  - if the planner's own output can't be parsed, it gets one bounded repair attempt (the parse error plus a size-capped snippet of the bad reply) before giving up with a clear error;
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
  └─ Run ───────────► POST /execute    → allow-listed runtimes → self-hosted Piston
```

The backend is FastAPI. It never executes submitted code itself: `/execute` checks the language and version against an allow-list (400 if not allowed), forwards the request to Piston, and maps failures to clear errors: 502 if Piston rejects the request, 503 if Piston is unreachable, 504 if it times out.

Under Docker Compose (see Setup below), the Compose services share an internal network: the backend reaches Piston at `http://piston:2000`, and the browser reaches the frontend container at `http://localhost:5173`, which in turn talks to the backend container directly. Piston itself is not exposed to the host.

---

## Setup

### Prerequisites
- Docker and Docker Compose (the `docker compose` CLI, Compose v2)
- A Groq API key (Tavily is optional)

### 1. Configure environment variables
From the repository root (the folder that contains `backend/` and `frontend/`):
```bash
cp backend/.env.example backend/.env
```
(On Windows cmd, use `copy backend\.env.example backend\.env` instead.)

Fill in `backend/.env`:

| Variable | Required? | Used by |
|---|---|---|
| `GROQ_API_KEY` | **Required** | Planner and developer agents, chat panel and code-change detection. |
| `GROQ_MODEL` | Optional | Model used by the planner and developer agents. Defaults to `openai/gpt-oss-20b`. The chat panel keeps its own model (`openai/gpt-oss-120b`). |
| `TAVILY_API_KEY` | Optional | Web search in the planner. Without it, the search returns no results and planning continues. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional | LangSmith tracing (off by default) — see below. |
| `AGENTCODE_MAX_PROMPT_TOKENS`, `AGENTCODE_MAX_429_WAIT_SECONDS` | Optional | Prompt-size guard and 429-retry tuning — see below. |

### 2. Start everything with Docker Compose
From the repository root:
```bash
docker compose up --build
```
This builds and starts four services:
- **piston** — the self-hosted Piston sandbox (pinned image digest), reachable only on the internal Compose network.
- **piston-init** — a one-shot service that installs the required Piston language packages and verifies all 7 supported runtimes are present, then exits; the backend waits for it to finish successfully before starting.
- **backend** and **frontend** — the FastAPI API and the built React app, served by Nginx.

Once `piston-init` reports success and the backend and frontend containers are up, open:
```
http://localhost:5173
```

To stop everything: `Ctrl+C`, then `docker compose down` (add `-v` only if you also want to drop the cached Piston language packages).

### Running the tests
The backend's automated tests run directly with Python, independent of Docker. In a terminal, from `backend/`:
```bash
python -m venv .venv
```
Activate it (Windows PowerShell: `.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`), then:
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

### Alternative: manual local setup (without Docker)
You can also run the backend and frontend directly, with Piston installed separately:

1. **Piston.** The public Piston API is not used. Run your own Piston container by following the [Piston README](https://github.com/engineer-man/piston), so that its API is reachable at `http://127.0.0.1:2000`, then install the runtimes listed under [Verification](#verification) with Piston's package manager. C++ compilation can exceed Piston's default compile timeout; if it does, raise it with `PISTON_LIMIT_OVERRIDES={"c++":{"compile_timeout":30000}}`.
2. **Backend.** From `backend/`, with a virtual environment active (see above): `pip install -r requirements.txt`, then `uvicorn main:app --host 127.0.0.1 --port 8000`.
3. **Frontend.** From `frontend/`: `npm install`, then `npm run dev`, and open `http://localhost:5173`. The backend's CORS setting allows only this origin.

### Optional: token usage and LangSmith tracing
- **Token usage** is always logged by the backend. Each LLM call writes a line like `LLM usage purpose=generate_steps model=... input_tokens=... output_tokens=... total_tokens=... source=groq-reported`, and each `/run-task` request ends with a `Task LLM usage: calls=... total_tokens=...` summary. `source=groq-reported` means the numbers came from Groq; `source=local-estimate` (about 4 characters per token) is used only if Groq reports no usage.
- **Rate limits (429):** the agent retries once only when Groq's suggested wait is short (`AGENTCODE_MAX_429_WAIT_SECONDS`, default 20); otherwise it fails at once with Groq's message.
- **Output-token limit and reasoning effort (GPT-OSS models):** Groq calls set an explicit completion-token ceiling (`MAX_OUTPUT_TOKENS`, 2048) and, for GPT-OSS models specifically, `reasoning_effort="low"` -- both in `backend/models/groq_llm.py`, not environment variables. This exists because GPT-OSS models on Groq default to `reasoning_effort="medium"`, which can spend most of an unconfigured output budget on internal reasoning and return an empty response (`finish_reason=length`); lowering reasoning effort and capping output length leaves more of the budget for the actual response.
- **Prompt-size guard:** `AGENTCODE_MAX_PROMPT_TOKENS` (default `0`, disabled). When set, prompts whose local estimate exceeds it are not sent.
- **LangSmith tracing** is off by default. To see the graph, each node's inputs and outputs, and each LLM call in [LangSmith](https://smith.langchain.com), set in `backend/.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=<your key>` and optionally `LANGSMITH_PROJECT=agentcode`, then restart the backend. Each task appears as a run named `agentcode_task`. Traces contain prompts and file contents, so only enable tracing for code you are happy to send to LangSmith.

---

## Verification

**Automated tests:** `106 passed, 2 skipped`. One skipped test needs permission to create symlinks on Windows; the other is the optional live Groq test (below). The tests cover:
- workspace confinement for the UI endpoints and the agent's file helpers;
- the `/execute` proxy and its error mapping;
- Groq API errors, timeouts and empty responses, with a bounded retry;
- token-usage logging (Groq-reported vs estimated), per-task totals, the 429 retry rule and the optional prompt-size guard;
- skipping the optional web search when the model returns no query;
- workflow error reporting;
- refusal of an agent edit that would leave broken Python;
- planner and developer parsing of typical model replies (fenced Python lists and JSON), through to the tool dispatcher;
- the planner's bounded repair retry: if its first reply fails to parse, it gets exactly one retry with the parse error and a size-capped snippet of the bad reply, before falling back to a clear error;
- the developer agent returning a planned `delete` step unchanged, without an LLM call, so it cannot be silently rewritten into a different tool.

The model calls are replaced with test doubles, so these tests make no API calls. An optional live test makes one real Groq call; it is skipped unless enabled (from `backend/`, PowerShell: `$env:AGENTCODE_LIVE_GROQ = "1"; python -m pytest -q tests/test_live_groq.py`).

**Docker Compose verification:**

| Check | Result |
|---|---|
| All 7 required Piston runtimes installed and verified by `piston-init` (python 3.12.0, javascript 18.15.0, typescript 5.0.3, java 15.0.2, csharp 6.12.0, c 10.2.0, c++ 10.2.0) | PASS |
| Backend, frontend and Piston integrated and reachable end-to-end through Docker Compose | PASS |
| AI agent Create, through the UI, at root level and inside a nested folder — verified against the real filesystem, not just the API response | PASS |
| AI agent Read, through the UI | PASS |
| AI agent Modify, through the UI, at root level and inside a nested folder — verified against the real filesystem | PASS |
| AI agent Delete, through the UI, at root level and inside a nested folder — verified against the real filesystem | PASS |
| Run button, through the UI, for Python | PASS |
| Run button, through the UI, for JavaScript | PASS |
| Run button, through the UI, for C++ | PASS |

These are manual, single-attempt runs against a non-deterministic model: they show the paths work, not that every request will succeed.

**Groq live run (token usage logging):** one UI task through the Groq-based agent, `Create a new file named usage_check.py containing print("usage check")`, succeeded and the file was written. The backend logged five LLM calls, all with `source=groq-reported`, and the per-call totals add up to the task summary:

| Node | Total tokens |
|---|---|
| `enhance_prompt` | 307 |
| `generate_search_query` | 495 |
| `decide_search_source` | 200 |
| `generate_steps` | 3,298 |
| `validate_step_with_llm` | 3,393 |
| **Task total** | **7,693** (6,863 input, 830 output) |

This is a single run. It shows that usage logging works; no token reduction has been quantified from it. Separately, the merge-sort task described below -- which had previously failed from GPT-OSS reasoning-token exhaustion followed by a 429 -- now succeeds after the reasoning-effort/output-token change, but this one success does not establish that Groq 429 errors have been eliminated generally; they remain possible for larger or multi-step tasks. The two codebase-aware calls (`generate_steps` and `validate_step_with_llm`) used most of the tokens, and the task total is close to the 8,000 tokens-per-minute limit we hit on Groq's free tier.

**Fresh-clone merge-sort run:** on an independent fresh clone (`D:\AgentCode`), a UI task asking the agent to write a merge sort implementation to `merge_sort.py` succeeded end to end: the agent generated the file, it was written to the workspace, and the generated code ran correctly through the Run button. This is qualitative confirmation that this task -- the same class of task that had previously failed during planning -- now completes; no token counts were recorded for this run, and it does not by itself quantify any reduction in token usage.

---

## Known limitations

- **Agent results depend on the model.** Plans and edits come from the Groq-hosted model, and a request can still be planned or applied incorrectly. There is no plan review before the agent edits files, and no automatic check that the result matches the request.
- **The UI sends the task as an object** (`{"text": ..., "id": ...}`) instead of plain text. The planner handled it in testing, but the payload has not been cleaned up.
- **Changing the Run dropdown's language does not change the editor's open filename or extension.** The dropdown selects which runtime `/execute` is called with; it does not rename the open file or switch its extension, so the file being run and the language selected can disagree if you change the dropdown without also opening or renaming a matching file.
- **No stdin.** The Run panel shows only the output text, not the exit code or signal.
- **No AI autocomplete.** The editor offers Monaco's built-in, non-AI suggestions only.
- **Execution is request/response**, not streamed: output appears when the run finishes.
- **Groq rate limits apply.** On the free tier, a task can hit HTTP 429 (tokens per minute). The agent retries once only when Groq suggests a short wait; otherwise it reports the error. The planner and developer still send full file contents, so larger tasks are the most likely to hit the limit. The reasoning-effort and output-token configuration (see [Optional: token usage and LangSmith tracing](#optional-token-usage-and-langsmith-tracing) above) reduces how much of a typical single-step call's budget goes to internal reasoning, which lowers the chance of hitting this limit for an ordinary task -- it does not raise or remove Groq's TPM ceiling, and a large or multi-step task can still hit it.
- **Long plans can fail.** No recursion limit is configured, so LangGraph's default limits how many developer steps a single task can run.
- **Not yet verified end to end:**
  - the explorer's own folder, rename and delete actions, as distinct from the AI agent's file operations (endpoints covered by unit tests only);
  - a live test of workspace confinement (unit tests only).
- **Some URLs remain hard-coded for a single-machine setup:** the backend's CORS allow-list is fixed to `http://localhost:5173`; its Piston URL falls back to `http://127.0.0.1:2000` only when no `PISTON_EXECUTE_URL` environment variable is set — Docker Compose always overrides this to the internal `piston:2000` address, so this only matters for the manual setup. Blocking calls run inside async endpoints, so a long agent task holds up other requests.

---

## Tech stack

- **Frontend:** React, Vite, Chakra UI, Monaco Editor
- **Backend:** FastAPI, httpx, Pydantic
- **Agents:** LangGraph (planner → developer); Groq via `langchain-groq`: `openai/gpt-oss-20b` by default for the agents (configurable with `GROQ_MODEL`), `openai/gpt-oss-120b` for chat
- **Code execution:** self-hosted Piston
- **Containerization:** Docker Compose (backend, frontend, Piston, and a one-shot `piston-init` runtime installer)
- **Tests:** pytest
