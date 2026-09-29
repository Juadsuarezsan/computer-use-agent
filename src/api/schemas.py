"""Pydantic models shared by the API, the agent loop and the eval harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActionType = Literal[
    "click",
    "double_click",
    "right_click",
    "mouse_move",
    "type",
    "key",
    "scroll",
    "screenshot",
    "wait",
    "task_complete",
]
"""Action vocabulary understood by every VM driver.

It is a deliberate subset of the Anthropic ``computer`` tool vocabulary:
``left_click`` maps to ``click``; ``left_click_drag`` / ``hold_key`` and other
exotic actions are rejected by the reasoner and replaced by ``screenshot``.
"""

ScrollDirection = Literal["up", "down", "left", "right"]

MAX_TASK_LENGTH = 2000
MAX_SCREEN_W = 4096
MAX_SCREEN_H = 4096


class Action(BaseModel):
    """One primitive action to perform on the virtual desktop.

    Attributes:
        type: Action kind (see :data:`ActionType`).
        coords: Absolute pixel coordinates for pointer actions.
        text: Text to type (``type``) or final answer (``task_complete``).
        key: Key combination in xdotool notation (``ctrl+a``, ``Return``).
        duration_ms: Wait duration for ``wait`` actions.
        scroll_direction: Direction for ``scroll`` actions.
        scroll_amount: Number of scroll "clicks" for ``scroll`` actions.
    """

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    coords: tuple[int, int] | None = None
    text: str | None = None
    key: str | None = None
    duration_ms: int = Field(default=0, ge=0, le=60_000)
    scroll_direction: ScrollDirection = "down"
    scroll_amount: int = Field(default=3, ge=0, le=100)

    @field_validator("coords")
    @classmethod
    def _coords_in_screen(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        """Reject negative or absurdly large coordinates (off-screen clicks)."""
        if value is None:
            return None
        x, y = value
        if x < 0 or y < 0 or x > MAX_SCREEN_W or y > MAX_SCREEN_H:
            raise ValueError(f"coordinates out of screen bounds: {value}")
        return value

    def summary(self) -> str:
        """Return a compact one-line description used in logs and traces."""
        if self.type in {"click", "double_click", "right_click", "mouse_move"}:
            return f"{self.type}{self.coords}"
        if self.type == "type":
            return f"type({self.text!r})"
        if self.type == "key":
            return f"key({self.key})"
        if self.type == "scroll":
            return f"scroll({self.scroll_direction}, {self.scroll_amount})"
        if self.type == "wait":
            return f"wait({self.duration_ms}ms)"
        if self.type == "task_complete":
            return f"task_complete({(self.text or '')[:60]!r})"
        return self.type


class Step(BaseModel):
    """Record of one iteration of the agent loop (audit-log friendly)."""

    step: int
    action: Action
    screenshot_path: str | None = None
    reasoning: str = ""
    execution_result: str = ""
    verifier_ok: bool = True
    verifier_note: str = ""
    safety_blocked: bool = False
    safety_reason: str | None = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class TaskRequest(BaseModel):
    """Body of ``POST /api/run-task``."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., min_length=2, max_length=MAX_TASK_LENGTH)
    task_id: str | None = Field(
        default=None,
        max_length=32,
        description="Optional id of a task from data/eval/tasks.json (runs on the sandbox app).",
    )
    max_steps: int | None = Field(default=None, ge=1, le=100)

    @field_validator("task")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only tasks."""
        if not value.strip():
            raise ValueError("task must not be blank")
        return value.strip()


class TaskResponse(BaseModel):
    """Result of one task execution."""

    trace_id: str = ""
    task: str
    task_id: str | None = None
    success: bool
    ground_truth_checked: bool = False
    steps: list[Step] = Field(default_factory=list)
    final_output: str = ""
    total_steps: int = 0
    blocked_actions: int = 0
    verifier_failures: int = 0
    recovered: bool = False
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    reasoner: str = ""
    vm: str = ""


class TaskSummary(BaseModel):
    """Public view of an eval task (``GET /api/tasks``)."""

    id: str
    category: str
    task: str
    app: str
    max_steps: int


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str
    version: str
    vm_mode: str
    llm_enabled: bool
    model: str
    tool_version: str
    extra: dict[str, Any] = Field(default_factory=dict)
