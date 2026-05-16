"""Decides the next action given the task + current screenshot description.

Production: Anthropic Claude with the `computer_20250124` tool.
Offline:    deterministic stub that walks a canned plan based on the task.
"""
from __future__ import annotations

from src.api.schemas import Action


class StubReasoner:
    """Deterministic plan for a few canonical tasks (offline tests)."""

    def __init__(self) -> None:
        self.history: list[Action] = []
        self._plan_index = 0

    def reset(self) -> None:
        self.history = []
        self._plan_index = 0

    def next_action(self, task: str, observation: str, step: int) -> tuple[Action, str]:
        t = task.lower()
        plan: list[Action]
        if "open" in t and ("file" in t or "folder" in t):
            plan = [
                Action(type="screenshot"),
                Action(type="click", coords=(50, 50)),
                Action(type="type", text="My Documents"),
                Action(type="key", key="Return"),
                Action(type="task_complete"),
            ]
        elif "csv" in t or "extract" in t:
            plan = [
                Action(type="screenshot"),
                Action(type="click", coords=(100, 200)),
                Action(type="key", key="ctrl+a"),
                Action(type="key", key="ctrl+c"),
                Action(type="type", text="data.csv"),
                Action(type="task_complete"),
            ]
        else:
            plan = [
                Action(type="screenshot"),
                Action(type="click", coords=(100, 100)),
                Action(type="task_complete"),
            ]
        idx = min(self._plan_index, len(plan) - 1)
        action = plan[idx]
        self._plan_index += 1
        self.history.append(action)
        reasoning = f"step {step}: planned {action.type}"
        return action, reasoning


class ClaudeComputerUseReasoner:
    """Uses Anthropic's Computer Use tool. Lazy-imported.

    Production code path. Requires `ANTHROPIC_API_KEY`. Each call sends the
    current screenshot + task + history; Claude returns the next action.
    """

    def __init__(self, model: str, api_key: str, tool_version: str) -> None:
        self.model = model
        self.api_key = api_key
        self.tool_version = tool_version
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history = []

    def next_action(self, task: str, observation: str, step: int) -> tuple[Action, str]:
        # The full Claude Computer Use loop sends image content blocks.
        # For now we return a screenshot-then-plan-complete envelope. Production
        # use: import anthropic; build messages with `tools=[{"type":"computer_20250124"...}]`
        # and parse `tool_use` blocks back into Action objects.
        if step == 0:
            return Action(type="screenshot"), "Initial screenshot capture"
        return Action(type="task_complete"), "Stub — wire Anthropic Computer Use tool here"
