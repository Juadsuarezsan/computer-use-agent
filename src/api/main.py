"""Computer Use Agent — FastAPI transport layer.

Endpoints:
    GET  /health              liveness + configuration summary
    GET  /api/tasks           the 20 eval tasks (id, category, text)
    GET  /api/safety/rules    the safety catalogue (blocked patterns, HITL rules)
    POST /api/run-task        run a task (ad-hoc text or a catalogue ``task_id``)
    GET  /api/audit/recent    last N task runs from the audit log
    GET  /api/eval/results    latest saved eval run(s)
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.orchestrator import run_task
from src.api.schemas import HealthResponse, TaskRequest, TaskResponse, TaskSummary
from src.config import REPO_ROOT, Settings, get_settings
from src.eval.tasks import default_tasks, get_task
from src.observability import configure_langsmith, configure_logging
from src.safety.blocklist import blocked_rules_catalog
from src.storage.audit import AuditLog, build_audit_log

APP_VERSION = "1.0.0"

limiter = Limiter(key_func=get_remote_address, default_limits=[get_settings().rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging/tracing and open the audit log for the process lifetime."""
    settings = get_settings()
    configure_logging()
    configure_langsmith(settings)
    app.state.audit = build_audit_log(settings)
    logger.info(
        f"api start vm_mode={settings.vm_mode} llm_enabled={settings.llm_enabled} "
        f"model={settings.anthropic_model} cors={settings.cors_origin_list}"
    )
    try:
        yield
    finally:
        app.state.audit.close()


app = FastAPI(
    title="Computer Use Agent",
    version=APP_VERSION,
    description="LangGraph loop driving a sandboxed browser/desktop with a safety pre-check.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


def _audit(request: Request) -> AuditLog:
    audit: AuditLog = request.app.state.audit
    return audit


SettingsDep = Depends(get_settings)
AuditDep = Depends(_audit)


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = SettingsDep) -> HealthResponse:
    """Liveness probe with the effective configuration (no secrets)."""
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        vm_mode=settings.vm_mode,
        llm_enabled=settings.llm_enabled,
        model=settings.anthropic_model,
        tool_version=settings.computer_use_tool,
        extra={"tasks": len(default_tasks()), "langsmith": settings.langchain_tracing_v2},
    )


@app.get("/api/tasks", response_model=list[TaskSummary])
async def list_tasks() -> list[TaskSummary]:
    """Return the eval catalogue (without plans or checkers)."""
    return [
        TaskSummary(id=t.id, category=t.category, task=t.task, app=t.app, max_steps=t.max_steps)
        for t in default_tasks()
    ]


@app.get("/api/safety/rules")
async def safety_rules() -> dict[str, list[str]]:
    """Return the safety catalogue used by ``docs/safety.md``."""
    return blocked_rules_catalog()


@app.post("/api/run-task", response_model=TaskResponse)
@limiter.limit(get_settings().rate_limit)
async def run(request: Request, req: TaskRequest, audit: AuditLog = AuditDep) -> TaskResponse:
    """Run a task through the agent loop on the configured VM.

    Validation errors return 422; an unknown ``task_id`` returns 404; any
    runtime failure is logged with its trace and returned as 500.
    """
    spec: dict[str, Any] | None = None
    if req.task_id:
        task = get_task(req.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"unknown task_id {req.task_id!r}")
        spec = task.as_spec()
    try:
        return await run_task(task=req.task, max_steps=req.max_steps, task_spec=spec, audit=audit)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("run_task failed")
        raise HTTPException(status_code=500, detail=f"agent failure: {exc}") from exc


@app.get("/api/audit/recent")
async def audit_recent(limit: int = 100, audit: AuditLog = AuditDep) -> list[dict[str, Any]]:
    """Return the most recent task runs (observability dashboard feed)."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    return audit.recent(limit)


@app.get("/api/eval/results")
async def eval_results() -> dict[str, Any]:
    """Return the saved eval runs (names + aggregates) without re-running anything."""
    runs_dir = REPO_ROOT / "eval" / "runs"
    out: dict[str, Any] = {"runs": []}
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out["runs"].append(
                {
                    "file": path.name,
                    "name": payload.get("name"),
                    "llm_used": payload.get("llm_used"),
                    "aggregate": payload.get("aggregate")
                    or {k: payload.get(k) for k in ("with_safety", "without_safety")},
                }
            )
    return out
