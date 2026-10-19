from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.planner import run_planner_subgraph as planner_agent
from agents.developer import run_developer_subgraph as developer_agent
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Main state class ---
class OverallState(TypedDict, total=False):
    task: str
    steps: list[dict]
    code: list[str]   # developer step logs
    status: str       # "planning_complete" | "development_complete" | "error"
    stage: str        # where a failure happened: "planner" | "developer"
    error: str


# --- Routing: only run the developer when the planner produced steps ---
def route_after_planner(state: OverallState) -> str:
    if state.get("status") == "planning_complete" and state.get("steps"):
        return "developer"
    return "end"


# --- Build Main LangGraph ---
def setup_graph():
    graph = StateGraph(OverallState)

    # --- Node 1: Planner ---
    def planner_node(state: OverallState) -> dict:
        logger.info(f"[Planner Node] Received task: {state['task']}")
        try:
            steps = planner_agent(state['task'])  # invokes the planner subgraph
        except Exception as e:
            # Keep the original exception and its traceback in the log
            logger.exception("Planner failed: %s", e)
            return {
                "status": "error",
                "stage": "planner",
                "error": f"{type(e).__name__}: {e}",
            }
        if not steps:
            logger.error("Planner produced no steps")
            return {
                "steps": [],
                "status": "error",
                "stage": "planner",
                "error": "Planner produced no steps",
            }
        return {
            "steps": steps,
            "status": "planning_complete"
        }

    # --- Node 2: Developer (only reached when the planner succeeded) ---
    def developer_node(state: OverallState) -> dict:
        logger.info(f"[Developer Node] Received steps: {state.get('steps')}")
        try:
            dev_result = developer_agent(state["steps"])  # invokes the developer subgraph
        except Exception as e:
            logger.exception("Developer failed: %s", e)
            return {
                "status": "error",
                "stage": "developer",
                "error": f"{type(e).__name__}: {e}",
            }
        failed = [log for log in dev_result if str(log).startswith("[ERROR]")]
        if failed:
            logger.error("Developer: %d of %d step(s) failed", len(failed), len(dev_result))
            return {
                "code": dev_result,
                "status": "error",
                "stage": "developer",
                "error": f"{len(failed)} of {len(dev_result)} step(s) failed: " + "; ".join(failed),
            }
        return {
            "code": dev_result,
            "status": "development_complete"
        }

    # Add nodes
    graph.add_node("planner_node", planner_node)
    graph.add_node("developer_node", developer_node)

    # Set transitions: a failed or empty plan ends the run without overwriting its error
    graph.set_entry_point("planner_node")
    graph.add_conditional_edges("planner_node", route_after_planner, {
        "developer": "developer_node",
        "end": END,
    })
    graph.add_edge("developer_node", END)

    return graph.compile()
