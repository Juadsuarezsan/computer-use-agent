"""VM executor — performs actions on the virtualized desktop (xdotool + scrot).

If `USE_FAKE_VM=true` or no VM is reachable, returns deterministic fake state
so the rest of the pipeline can be tested.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.api.schemas import Action


@dataclass
class VMState:
    screenshot_path: str
    last_observation: str
    step: int = 0


class FakeVM:
    """Deterministic VM for tests + offline demos."""

    def __init__(self) -> None:
        self.step = 0
        self.events: list[str] = []

    async def screenshot(self) -> tuple[str, str]:
        self.step += 1
        return (
            f"/tmp/fake_screenshot_{self.step}.png",
            f"Step {self.step}: desktop with file manager open, no errors visible.",
        )

    async def execute(self, action: Action) -> str:
        self.events.append(f"{action.type}:{action.text or action.coords or action.key or ''}")
        if action.type == "click":
            return f"Clicked at {action.coords}"
        if action.type == "type":
            return f"Typed: {action.text!r}"
        if action.type == "key":
            return f"Pressed: {action.key}"
        if action.type == "scroll":
            return f"Scrolled {action.duration_ms}ms"
        if action.type == "wait":
            return f"Waited {action.duration_ms}ms"
        if action.type == "task_complete":
            return "Marked task complete."
        return f"Action: {action.type}"


class XdoToolVM:
    """Real VM driver — uses xdotool + scrot. Lazy-imported to avoid hard dep in CI."""

    def __init__(self) -> None:
        import subprocess
        self.subprocess = subprocess
        self.step = 0

    async def screenshot(self) -> tuple[str, str]:
        self.step += 1
        path = f"/tmp/cu_screenshot_{self.step}.png"
        self.subprocess.run(["scrot", path], check=False)
        return path, f"Step {self.step}: see {path}"

    async def execute(self, action: Action) -> str:
        if action.type == "click" and action.coords:
            x, y = action.coords
            self.subprocess.run(["xdotool", "mousemove", str(x), str(y), "click", "1"], check=False)
            return f"Clicked at {x},{y}"
        if action.type == "type" and action.text:
            self.subprocess.run(["xdotool", "type", "--delay", "20", action.text], check=False)
            return f"Typed: {action.text}"
        if action.type == "key" and action.key:
            self.subprocess.run(["xdotool", "key", action.key], check=False)
            return f"Pressed: {action.key}"
        if action.type == "scroll":
            self.subprocess.run(["xdotool", "click", "5"], check=False)
            return "Scrolled"
        return f"Action: {action.type}"


def build_vm():
    if os.environ.get("USE_FAKE_VM", "true").lower() == "true":
        return FakeVM()
    try:
        return XdoToolVM()
    except Exception:
        return FakeVM()
