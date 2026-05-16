"""Computer Use Agent API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from src.agent.orchestrator import run_task
from src.api.schemas import TaskRequest, TaskResponse
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Computer Use Agent",
    version="0.5.0",
    description="LangGraph loop driving a virtualized desktop with safety pre-check.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    import os
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.5.0",
        "stage": "substantive",
        "vm_mode": "fake" if os.environ.get("USE_FAKE_VM", "true").lower() == "true" else "xdotool",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/run-task", response_model=TaskResponse)
async def run(req: TaskRequest) -> TaskResponse:
    try:
        return await run_task(task=req.task, max_steps=req.max_steps)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_task failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eval/run")
async def eval_endpoint() -> dict:
    from src.eval.runner import run_eval
    return await run_eval()
