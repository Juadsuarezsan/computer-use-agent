"""Playwright + Chromium sandbox: a reproducible "VM" that needs no Docker.

The fixture web apps in ``sandbox/webapps/`` are loaded from ``file://`` URLs
in a headless Chromium. Actions are executed with real mouse/keyboard events
at pixel coordinates (exactly what the model sees), screenshots are real PNGs,
and ground truth is checked deterministically against the DOM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.protocols import Observation, UIElement
from src.api.schemas import Action
from src.sandbox.checkers import evaluate_checker

ELEMENT_MAP_JS = """
() => {
  const interactive = 'a[href], button, input, select, textarea, [role="button"], [role="link"], [data-target]';
  const out = [];
  const vw = window.innerWidth, vh = window.innerHeight;
  for (const el of document.querySelectorAll(interactive)) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx > vw || cy > vh) continue;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;
    if (el.disabled) continue;
    const tag = el.tagName.toLowerCase();
    let role = el.getAttribute('role') || tag;
    if (tag === 'input') role = el.type === 'checkbox' || el.type === 'radio' ? el.type : 'textbox';
    if (tag === 'a') role = 'link';
    let selector = el.id ? '#' + el.id : (el.getAttribute('data-target') ? '[data-target="' + el.getAttribute('data-target') + '"]' : '');
    if (!selector && el.name) selector = tag + '[name="' + el.name + '"]';
    if (!selector) continue;
    const label = (el.getAttribute('aria-label') || el.placeholder || (el.labels && el.labels[0] ? el.labels[0].innerText : '') || el.innerText || el.value || el.name || '').trim().slice(0, 60);
    out.push({selector, role, label, x: Math.round(cx), y: Math.round(cy)});
  }
  const fields = {};
  for (const el of document.querySelectorAll('[data-field]')) {
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (r.width < 1 || style.display === 'none' || style.visibility === 'hidden') continue;
    fields[el.getAttribute('data-field')] = (el.innerText || el.value || '').trim();
  }
  return {elements: out, fields, title: document.title, hash: location.hash};
}
"""

KEY_MAP = {
    "return": "Enter",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "space": "Space",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "home": "Home",
    "end": "End",
    "page_up": "PageUp",
    "page_down": "PageDown",
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "shift": "Shift",
    "super": "Meta",
    "meta": "Meta",
    "cmd": "Meta",
}


def xdotool_key_to_playwright(key: str) -> str:
    """Translate xdotool key syntax (``ctrl+a``, ``Return``) to Playwright's."""
    parts = [p for p in key.replace(" ", "").split("+") if p]
    out: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered in KEY_MAP:
            out.append(KEY_MAP[lowered])
        elif len(part) == 1:
            out.append(part)
        elif lowered.startswith("f") and lowered[1:].isdigit():
            out.append(part.upper())
        else:
            out.append(part[0].upper() + part[1:])
    return "+".join(out)


class PlaywrightVM:
    """Headless-Chromium sandbox implementing the :class:`~src.agent.protocols.VM` protocol.

    Args:
        chromium_path: Chromium executable (``/opt/pw-browsers/chromium`` here).
        webapps_dir: Directory with the fixture HTML apps.
        screenshot_dir: Where PNGs are stored (one per step).
        width / height: Viewport size (must match the model's display size).
        headless: Run without a window (always ``True`` in CI).
    """

    name = "playwright"

    def __init__(
        self,
        *,
        chromium_path: str | None = None,
        webapps_dir: str,
        screenshot_dir: str = "data/screenshots",
        width: int = 1024,
        height: int = 768,
        headless: bool = True,
    ) -> None:
        self.chromium_path = chromium_path
        self.webapps_dir = Path(webapps_dir)
        self.screenshot_dir = Path(screenshot_dir)
        self.width = width
        self.height = height
        self.headless = headless
        self.step = 0
        self.task_id = "adhoc"
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None

    async def _ensure_browser(self) -> Any:
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        launch_kwargs: dict[str, Any] = {"headless": self.headless}
        if self.chromium_path:
            launch_kwargs["executable_path"] = self.chromium_path
        self._browser = await self._pw.chromium.launch(**launch_kwargs)
        context = await self._browser.new_context(
            viewport={"width": self.width, "height": self.height}, device_scale_factor=1
        )
        self._page = await context.new_page()
        logger.info(f"Playwright sandbox started ({self.width}x{self.height})")
        return self._page

    def app_url(self, app: str, start: str = "") -> str:
        """Return the ``file://`` URL of a fixture app plus optional ``#route``."""
        path = (self.webapps_dir / app).resolve()
        if not path.exists():
            raise FileNotFoundError(f"fixture app not found: {path}")
        return path.as_uri() + (start if start.startswith("#") or start.startswith("?") else "")

    async def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Load the task's fixture app at its start route."""
        page = await self._ensure_browser()
        spec = spec or {}
        self.step = 0
        self.task_id = str(spec.get("id", "adhoc"))
        app = str(spec.get("app", "dashboard.html"))
        # about:blank first so a same-document hash change cannot keep old JS state
        await page.goto("about:blank")
        await page.goto(self.app_url(app, str(spec.get("start", ""))), wait_until="load")
        await page.evaluate("() => { if (window.__sandboxReset) window.__sandboxReset(); }")
        await page.wait_for_timeout(50)

    async def screenshot(self) -> Observation:
        """Capture a PNG and the interactive element map."""
        page = await self._ensure_browser()
        self.step += 1
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"{self.task_id}_{self.step:03d}.png"
        png: bytes = await page.screenshot(path=str(path), type="png")
        info = await page.evaluate(ELEMENT_MAP_JS)
        elements = [
            UIElement(
                selector=str(e["selector"]),
                role=str(e["role"]),
                label=str(e["label"]),
                x=int(e["x"]),
                y=int(e["y"]),
            )
            for e in info["elements"]
        ]
        listing = "; ".join(f"{e.role} '{e.label}' @({e.x},{e.y})" for e in elements[:40])
        description = f"{info['title']} {info['hash']} | {len(elements)} interactive elements: {listing}"
        return Observation(
            screenshot_path=str(path),
            description=description,
            png=png,
            elements=elements,
            fields={str(k): str(v) for k, v in info["fields"].items()},
            url=page.url,
        )

    async def execute(self, action: Action) -> str:
        """Perform the action with real input events."""
        page = await self._ensure_browser()
        if action.type in {"click", "double_click", "right_click"} and action.coords:
            x, y = action.coords
            kwargs: dict[str, Any] = {}
            if action.type == "double_click":
                kwargs["click_count"] = 2
            if action.type == "right_click":
                kwargs["button"] = "right"
            await page.mouse.click(x, y, **kwargs)
            await page.wait_for_timeout(30)
            return f"{action.type} at ({x},{y})"
        if action.type == "mouse_move" and action.coords:
            await page.mouse.move(*action.coords)
            return f"moved to {action.coords}"
        if action.type == "type" and action.text:
            await page.keyboard.type(action.text, delay=5)
            return f"typed {len(action.text)} chars"
        if action.type == "key" and action.key:
            combo = xdotool_key_to_playwright(action.key)
            await page.keyboard.press(combo)
            await page.wait_for_timeout(30)
            return f"pressed {combo}"
        if action.type == "scroll":
            if action.coords:
                await page.mouse.move(*action.coords)
            dx = dy = 0
            delta = action.scroll_amount * 100
            if action.scroll_direction == "down":
                dy = delta
            elif action.scroll_direction == "up":
                dy = -delta
            elif action.scroll_direction == "right":
                dx = delta
            else:
                dx = -delta
            await page.mouse.wheel(dx, dy)
            await page.wait_for_timeout(30)
            return f"scrolled {action.scroll_direction} x{action.scroll_amount}"
        if action.type == "wait":
            await page.wait_for_timeout(action.duration_ms)
            return f"waited {action.duration_ms}ms"
        if action.type == "task_complete":
            return "task marked complete"
        return f"no-op: {action.type}"

    async def check(self, checker: dict[str, Any], answer: str | None) -> bool:
        """Evaluate a ground-truth checker against the live DOM."""
        page = await self._ensure_browser()
        return await evaluate_checker(page, checker, answer)

    async def close(self) -> None:
        """Shut the browser down."""
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._browser = self._page = None
