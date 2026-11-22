import re
import ast
import json
import argparse
import requests
import logging
import os
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field, ValidationError
from models.groq_llm import chat, EmptyResponseError
from utils.file_ops import list_code_files, read_code_file
from langgraph.graph import StateGraph, END
import time
import os
from dotenv import load_dotenv

load_dotenv()


# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tavily search helper ---
def tavily_search(query: str, max_results: int = 5, retries: int = 3) -> list:
    api_key = os.getenv("TAVILY_API_KEY")
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_images": False,
        "max_results": max_results
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Tavily response: {data}")
            logger.info(f"Tavily returned {len(data.get('results', []))} results.")
            return data.get("results", [])
        except requests.exceptions.Timeout:
            logger.warning(f"Tavily timeout, retrying ({attempt+1}/{retries})...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            break
    return []

# --- ToolStep schema ---
class ToolStep(BaseModel):
    file: str = Field(..., description="Filename to operate on")
    tool: str = Field(..., description="One of ['read','write','delete','apply_change','llm_modify']")
    args: dict = Field(default_factory=dict, description="Arguments for the tool operation")

# --- LangGraph State Definition ---
class PlannerState(TypedDict, total=False):
    task: str
    codebase_summary: str
    enhanced_task: str
    use_external: bool
    search_query: str
    external_results: List[dict]
    steps: List[ToolStep]

# --- Node Functions ---
def summarize_codebase(state: PlannerState) -> PlannerState:
    logger.info("[Node] summarize_codebase")
    try:
        files = list_code_files()
    except Exception:
        files = []
    summary = []
    for f in files:
        try:
            content = ''.join(read_code_file(f))
            summary.append(f"---\nFilename: {f}\n{content}\n")
        except Exception:
            summary.append(f"---\nFilename: {f}\n[Unreadable]\n")
    logger.info(f"Summarized {len(summary)} files.")
    return {"codebase_summary": "".join(summary)}

def enhance_prompt(state: PlannerState) -> PlannerState:
    logger.info("[Node] enhance_prompt")
    # Only the task is sent here; the full codebase is used later, in generate_steps.
    prompt = (
        f"Enhance this task for clarity: {state['task']} Do not give any code just imrpove the task language"
        f"Just give the enhanced task nothing else no additional information"
    )
    enhanced = chat(prompt)
    logger.info(f"Enhanced task: {enhanced.strip()}")
    return {"enhanced_task": enhanced}

def generate_search_query(state: PlannerState) -> PlannerState:
    logger.info("[Node] generate_search_query")
    prompt = (
        f"You are an expert research assistant. Given the following enhanced task, "
        f"generate a single, concise web search query (or a comma-separated list of queries) for any EXTRA information required to complete the task like some information or article or api link etc."
        f"that would help a developer complete the task. "
        f"Do NOT include explanations, just the query or queries. do not mention any year\n\n "
        f"Enhanced Task:\n{state['enhanced_task']}\n"
    )
    try:
        query = chat(prompt).strip()
    except EmptyResponseError as e:
        # The web search is optional context: an empty model answer means "no query",
        # so planning continues without external search. Other LLM errors still propagate.
        logger.warning(f"No search query returned ({e}); continuing without external search")
        return {"search_query": ""}
    logger.info(f"Generated search query: {query}")
    return {"search_query": query}

def decide_search_source(state: PlannerState) -> PlannerState:
    logger.info("[Node] decide_search_source")
    if not state.get("search_query"):
        logger.info("No search query: skipping external search")
        return {"use_external": False}
    # Decided from the task text only; the full codebase is not needed for this yes/no choice.
    prompt = (
        f"Given the task, should we fetch external docs?\n"
        f"Task: {state['task']}\nEnhanced task: {state.get('enhanced_task', '')}\n"
        f"Answer with EXTERNAL or INTERNAL."
    )
    result = chat(prompt).strip().upper()
    logger.info(f"Decision from LLM: {result}")
    return {"use_external": result == "EXTERNAL"}

def search_external(state: PlannerState) -> PlannerState:
    logger.info("[Node] search_external")
    logger.info(f"State keys at search_external: {list(state.keys())}")
    query = state.get("search_query", state["task"])
    logger.info(f"Query passed to Tavily: {query}")
    results = tavily_search(query)
    return {"external_results": results}

def _strip_markdown_fence(text: str) -> str:
    """Strip a single markdown code fence that wraps the ENTIRE response.

    Supports ```json, ```python, and bare ``` fences, with LF or CRLF line
    endings. Only removes a fence when both an opening fence (at the very
    start) and a closing fence (at the very end) are present -- it never
    strips a ``` sequence that appears inside the body of the response, so
    a triple-backtick sequence inside generated file content is preserved.
    """
    stripped = text.strip()
    opening = re.match(r"^```[a-zA-Z]*\r?\n", stripped)
    closing = re.search(r"\r?\n```$", stripped)
    if opening and closing and closing.start() >= opening.end():
        return stripped[opening.end():closing.start()].strip()
    return stripped

def generate_steps(state: PlannerState) -> PlannerState:
    logger.info("[Node] generate_steps")
    external_results = state.get('external_results', [])
    external_text = ""
    if external_results:
        external_text = "\n\nExternal Web Search Results:\n"
        for i, res in enumerate(external_results, 1):
            external_text += f"{i}. {res.get('title', '')}\nURL: {res.get('url', '')}\n{res.get('content', '')}\n\n"

    prompt = f"""
You are a software planning agent. Given the following enhanced task, codebase, and web search results, break it down into the smallest possible, atomic steps.
Generate a python list of steps , Each step must mention:
- "file": the filename to operate on (specify the correct extension for the language, e.g., .js, .html, .py, .css, .java, etc.)
- "tool": one of ["read", "write", "delete", "apply_change", "llm_modify"]
- "args": a dict of arguments for the tool (e.g., code, line number, function name, etc.)

"write" replaces the ENTIRE file. To change an existing file with "write", its "content" arg must be the complete updated file, including all existing code that the task does not ask to remove.

Creating a new file is itself an actionable step: if the task asks you to create or save a file, that is enough information to act on, and you must produce a "write" step for it rather than deciding there is nothing to do.

The "file" value may include a relative subdirectory path (e.g. "folder_name/example.py"), including a directory that already exists in the codebase. Naming it as part of "file" is sufficient when that directory is already present; you do not need a separate tool to create it first.

The codebase shown above only lists top-level files and may not include files inside subdirectories, even ones the task describes as already existing -- trust the task's description of existing folders rather than requiring confirmation from the codebase listing, and do not treat writing the one requested file inside such a folder as creating an 'additional' file or directory.

For a simple file-creation request, output exactly one "write" step containing the requested filename (with its subdirectory path, if any) and the complete file content. For example, given the task "Create a new file named notes.txt inside the existing folder docker_agent_smoke_test, containing the line 'hello'", the correct output is:

[{{"file": "docker_agent_smoke_test/notes.txt", "tool": "write", "args": {{"content": "hello\\n"}}}}]

Only output an empty list, [], when there is genuinely no actionable file operation in the task (for example, the task asks a question rather than requesting any file change) -- never because the requested path includes a subdirectory, or because no explicit "create directory" tool exists.

Choose the file type and language that best fits the task. If the task involves web, use .html, .js, .css as needed. If the task is for Python, use .py. If the codebase is empty, create all necessary files from scratch.

Be specific: reference real files, functions, and lines. Do not use placeholders like '...' or generic names. If a file does not exist, create it. If you need to add a function or code to an existing file, use "write" with the complete updated file (the existing code plus the addition). If you need to modify a line, specify the line number and the new code.

Output ONLY a valid Python list of steps, no explanations, no markdown, no extra text.

Codebase:
{state['codebase_summary']}

Enhanced Task:
{state['enhanced_task']}
If the required information to complete the task is present in this {external_text} include them in the result.


If you cannot determine the steps, output an empty list: []


"""
    response = chat(prompt)
    clean = _strip_markdown_fence(response)
    try:
        steps_raw = ast.literal_eval(clean)
    except Exception as e_literal:
        try:
            steps_raw = json.loads(clean)
        except Exception as e_json:
            snippet = clean if len(clean) <= 500 else clean[:500] + "... [truncated]"
            logger.error(
                f"Failed to parse planner steps. ast.literal_eval error: {e_literal}; "
                f"json.loads error: {e_json}\nRaw string was:\n{snippet}"
            )
            raise RuntimeError(
                f"Could not parse planner output as steps (ast.literal_eval: {e_literal}; json.loads: {e_json})"
            ) from e_json

    steps: List[ToolStep] = []
    for item in steps_raw:
        try:
            steps.append(ToolStep(**item))
        except ValidationError as e:
            raise RuntimeError(f"Step validation failed: {e}")
    if not steps:
        logger.warning(
            "Planner generated zero steps. "
            "Parsed items: %d. Raw model response: %s",
            len(steps_raw),
            clean[:2000],
        )
    logger.info(f"Generated {len(steps)} steps.")
    return {"steps": steps}

# --- Build LangGraph ---
builder = StateGraph(PlannerState)

builder.add_node("summarize_codebase", summarize_codebase)
builder.add_node("enhance_prompt", enhance_prompt)
builder.add_node("generate_search_query", generate_search_query)
builder.add_node("decide_search_source", decide_search_source)
builder.add_node("search_external", search_external)
builder.add_node("generate_steps", generate_steps)

# Edges
builder.set_entry_point("summarize_codebase")
builder.add_edge("summarize_codebase", "enhance_prompt")
builder.add_edge("enhance_prompt", "generate_search_query")
builder.add_edge("generate_search_query", "decide_search_source")

builder.add_conditional_edges(
    "decide_search_source",
    lambda state: "external" if state["use_external"] else "internal",
    {
        "external": "search_external",
        "internal": "generate_steps"
    }
)

builder.add_edge("search_external", "generate_steps")
builder.add_edge("generate_steps", END)

graph = builder.compile()

# --- Public Runner ---
def run_planner_subgraph(task: str) -> List[dict]:
    logger.info(f"Running planner for task: {task}")
    output = graph.invoke({"task": task})
    logger.info("Planner execution completed.")
    return [step.model_dump() for step in output["steps"]]

# --- CLI Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="Task to plan", required=False)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test_tasks = [
            "Add a health-check endpoint to server.py",
            "Refactor the database connector into a new module",
            "Enhance existing authentication flow to use OAuth2"
        ]
        for task in test_tasks:
            print(f"\n=== Task: {task} ===")
            result = run_planner_subgraph(task)
            print(json.dumps(result, indent=2))
    elif args.task:
        result = run_planner_subgraph(args.task)
        print(json.dumps(result, indent=2))
    else:
        parser.error("Provide --task or --test")
