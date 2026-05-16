from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ActionType = Literal["click", "type", "key", "scroll", "screenshot", "wait", "task_complete"]


class Action(BaseModel):
    type: ActionType
    coords: tuple[int, int] | None = None
    text: str | None = None
    key: str | None = None
    duration_ms: int = 0


class Step(BaseModel):
    step: int
    action: Action
    screenshot_path: str | None = None
    reasoning: str = ""
    verifier_ok: bool = True
    safety_blocked: bool = False
    safety_reason: str | None = None


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=2)
    max_steps: int | None = None


class TaskResponse(BaseModel):
    task: str
    success: bool
    steps: list[Step] = Field(default_factory=list)
    final_output: str = ""
    total_steps: int = 0
    blocked_actions: int = 0
    latency_ms: int = 0
