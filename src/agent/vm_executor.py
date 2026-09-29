"""VM drivers that do not need a browser: the deterministic fake and xdotool.

* :class:`FakeVM` — in-memory, used by unit tests and CI.
* :class:`XdoToolVM` — drives a real X display (Xvfb in the sandbox image)
  with ``scrot`` + ``xdotool`` through ``subprocess`` (mocked in tests).
* :func:`build_vm` — factory selecting the driver from ``Settings.vm_mode``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.protocols import Observation, UIElement
from src.api.schemas import Action
from src.config import Settings, get_settings


class FakeVM:
    """Deterministic VM: no screen, but a synthetic element map and fields.

    The fake exposes a fixed set of elements so scripted plans and the
    orchestrator can be exercised without Playwright.
    """

    name = "fake"

    def __init__(self) -> None:
        self.step = 0
        self.events: list[str] = []
        self.typed: list[str] = []
        self.fields: dict[str, str] = {"version": "0.0-fake"}
        self.url = "fake://desktop"

    async def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Clear counters and events."""
        self.step = 0
        self.events = []
        self.typed = []
        self.url = f"fake://{(spec or {}).get('app', 'desktop')}"

    async def screenshot(self) -> Observation:
        """Return a synthetic observation whose description changes every step."""
        self.step += 1
        elements = [
            UIElement("#ok", "button", "OK", 100, 100),
            UIElement("#input", "textbox", "Input", 300, 200),
        ]
        return Observation(
            screenshot_path=f"fake_screenshot_{self.step}.png",
            description=f"Step {self.step}: desktop with file manager open, no errors visible.",
            png=None,
            elements=elements,
            fields=dict(self.fields),
            url=self.url,
        )

    async def execute(self, action: Action) -> str:
        """Record the action and return a description."""
        self.events.append(action.summary())
        if action.type == "type" and action.text:
            self.typed.append(action.text)
            return f"Typed: {action.text!r}"
        if action.type in {"click", "double_click", "right_click"}:
            return f"{action.type} at {action.coords}"
        if action.type == "key":
            return f"Pressed: {action.key}"
        if action.type == "scroll":
            return f"Scrolled {action.scroll_direction} x{action.scroll_amount}"
        if action.type == "wait":
            return f"Waited {action.duration_ms}ms"
        if action.type == "task_complete":
            return "Marked task complete."
        return f"Action: {action.type}"

    async def check(self, checker: dict[str, Any], answer: str | None) -> bool:
        """The fake VM cannot verify DOM ground truth: only answer checkers are supported."""
        from src.sandbox.checkers import check_answer_only

        return check_answer_only(checker, answer)

    async def close(self) -> None:
        """Nothing to release."""


KEY_ALIASES = {"enter": "Return", "esc": "Escape"}


class XdoToolVM:
    """Real desktop driver using ``scrot`` + ``xdotool`` on an X display.

    Args:
        screenshot_dir: Directory for PNG captures.
        runner: Callable with the ``subprocess.run`` signature (injectable for tests).
    """

    name = "xdotool"

    def __init__(self, screenshot_dir: str = "data/screenshots", runner: Any = subprocess.run) -> None:
        for tool in ("xdotool", "scrot"):
            if shutil.which(tool) is None and runner is subprocess.run:
                raise RuntimeError(f"{tool} is not installed; use VM_MODE=playwright or fake")
        self._run = runner
        self.step = 0
        self.screenshot_dir = Path(screenshot_dir)

    def _cmd(self, args: list[str]) -> str:
        proc = self._run(args, check=False, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            logger.warning(f"{args[0]} exited {proc.returncode}: {proc.stderr.strip()[:200]}")
            return f"{args[0]} failed (rc={proc.returncode})"
        return "ok"

    async def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Nothing to reset on a real desktop (the operator prepares it)."""
        self.step = 0

    async def screenshot(self) -> Observation:
        """Capture the screen with ``scrot``."""
        self.step += 1
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"xdo_{self.step:03d}.png"
        self._cmd(["scrot", "--overwrite", str(path)])
        png = path.read_bytes() if path.exists() else None
        return Observation(
            screenshot_path=str(path),
            description=f"Step {self.step}: X display capture {path.name}",
            png=png,
            url="x11://display",
        )

    async def execute(self, action: Action) -> str:
        """Translate the action into xdotool calls."""
        if action.type in {"click", "double_click", "right_click"} and action.coords:
            x, y = action.coords
            button = "3" if action.type == "right_click" else "1"
            repeat = ["--repeat", "2"] if action.type == "double_click" else []
            return self._cmd(["xdotool", "mousemove", str(x), str(y), "click", *repeat, button])
        if action.type == "mouse_move" and action.coords:
            x, y = action.coords
            return self._cmd(["xdotool", "mousemove", str(x), str(y)])
        if action.type == "type" and action.text:
            return self._cmd(["xdotool", "type", "--delay", "20", action.text])
        if action.type == "key" and action.key:
            key = KEY_ALIASES.get(action.key.lower(), action.key)
            return self._cmd(["xdotool", "key", key])
        if action.type == "scroll":
            button = {"up": "4", "down": "5", "left": "6", "right": "7"}[action.scroll_direction]
            return self._cmd(["xdotool", "click", "--repeat", str(action.scroll_amount), button])
        if action.type == "wait":
            await asyncio.sleep(action.duration_ms / 1000)
            return f"Waited {action.duration_ms}ms"
        return f"Action: {action.type}"

    async def check(self, checker: dict[str, Any], answer: str | None) -> bool:
        """A raw X display has no DOM: only answer checkers apply."""
        from src.sandbox.checkers import check_answer_only

        return check_answer_only(checker, answer)

    async def close(self) -> None:
        """Nothing to release."""


def build_vm(settings: Settings | None = None) -> Any:
    """Build the VM driver selected by ``VM_MODE``.

    Raises:
        RuntimeError: when the requested driver cannot be constructed. There is
            no silent fallback: a misconfigured production VM must be visible.
    """
    s = settings or get_settings()
    if s.vm_mode == "fake":
        return FakeVM()
    if s.vm_mode == "xdotool":
        return XdoToolVM(screenshot_dir=s.screenshot_dir)
    from src.agent.playwright_vm import PlaywrightVM

    return PlaywrightVM(
        chromium_path=s.chromium_path,
        webapps_dir=s.webapps_dir,
        screenshot_dir=s.screenshot_dir,
        width=s.display_width,
        height=s.display_height,
    )
