"""Computer Use Agent — placeholder until v0.1.0 build out."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Computer Use Agent",
    version="0.1.0",
    description="Computer Use Agent — Anthropic Computer Use API + VM",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "stage": "scaffolding",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }

class TaskRequest(BaseModel):
    task: str
    max_steps: int | None = None


class TaskStep(BaseModel):
    step: int
    action: str  # click(x,y) | type(text) | key(combo) | scroll
    screenshot_path: str | None = None
    reasoning: str | None = None


class TaskResponse(BaseModel):
    task: str
    success: bool
    steps: list[TaskStep] = []
    final_output: str = ""
    total_steps: int = 0
    latency_ms: int = 0


@app.post("/api/run-task", response_model=TaskResponse)
async def run_task(req: TaskRequest) -> TaskResponse:
    return TaskResponse(task=req.task, success=False, steps=[], final_output="not_yet_implemented")
