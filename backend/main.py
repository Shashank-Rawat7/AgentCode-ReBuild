import httpx
from fastapi import HTTPException
from pydantic import Field
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.chat_agent import chat_with_memory, memory ,chat_with_memory_with_confirmation # Import your chat agent
from langgraph_app.graph import setup_graph
from utils.file_ops import router as file_ops_router
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


app = FastAPI()

# Allow CORS for your frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(file_ops_router)
PISTON_EXECUTE_URL = "http://127.0.0.1:2000/api/v2/execute"

ALLOWED_RUNTIMES = {
    "python": {"3.12.0"},
    "javascript": {"18.15.0"},
    "typescript": {"5.0.3"},
    "java": {"15.0.2"},
    "csharp": {"6.12.0"},
    "c": {"10.2.0"},
    "cpp": {"10.2.0"},
}

class ExecuteFile(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content: str = Field(max_length=100_000)


class ExecuteRequest(BaseModel):
    language: str = Field(min_length=1, max_length=50)
    version: str = Field(min_length=1, max_length=50)
    files: list[ExecuteFile] = Field(min_length=1, max_length=10)


@app.post("/execute")
async def execute_code(req: ExecuteRequest):
    if req.language not in ALLOWED_RUNTIMES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if req.version not in ALLOWED_RUNTIMES[req.language]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported version for this language",
        )
    payload = req.model_dump()

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(PISTON_EXECUTE_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Piston execution request timed out")
    except httpx.HTTPStatusError as exc:
        try:
            piston_error = exc.response.json()
        except ValueError:
            piston_error = {}

        detail = piston_error.get("message") or piston_error.get("error")

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Piston rejected the execution request",
                "piston_status": exc.response.status_code,
                "piston_error": str(detail)[:500] if detail else None,
            },
        )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Local Piston service is unavailable")

class ChatRequest(BaseModel): 
    message: str
    custom_instructions: str = "Behave like a helpful assitant please"  # Default instructions

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    reply = chat_with_memory_with_confirmation(req.message, req.custom_instructions)
    return reply

@app.post("/reset")
async def reset_endpoint():
    memory.clear()
    return {"status": "memory reset"}




@app.post("/run-task")
async def run_task_endpoint(data: dict = Body(...)):
    print("run-task called with:", data)
    task = data.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="No task provided")
    try:
        workflow = setup_graph()
        result = workflow.invoke({"task": task})
    except Exception as e:
        logger.exception("Workflow failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "stage": "workflow", "error": f"{type(e).__name__}: {e}", "logs": []},
        )

    logs = result.get("code") or []
    if result.get("status") == "development_complete":
        return {"status": "success", "steps": result.get("steps", []), "logs": logs}

    # Planner or developer reported a failure: surface it instead of claiming success
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "stage": result.get("stage", "unknown"),
            "error": result.get("error") or "Workflow did not complete",
            "logs": logs,
        },
    )
