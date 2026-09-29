"""Reasoners: decide the next :class:`~src.api.schemas.Action` from an observation.

* :class:`StubReasoner` — offline, deterministic. Follows a *scripted plan*
  from ``data/eval/tasks.json`` whose targets are CSS selectors resolved
  against the DOM element map of the current observation (a DOM-oracle
  baseline: it measures the sandbox, executor and checkers, **not** the
  intelligence of a model). Without a plan it falls back to keyword plans.
* :class:`ClaudeComputerUseReasoner` — production path. Sends the screenshot
  as an ``image`` block with the Anthropic ``computer`` tool and parses the
  returned ``tool_use`` block into an :class:`Action`.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agent.llm import LLMClient, content_to_params, image_block, text_block
from src.agent.protocols import Observation, ReasonerOutput
from src.api.schemas import Action

SYSTEM_PROMPT = (
    "You are an autonomous computer-use agent operating a sandboxed browser at "
    "{width}x{height} pixels. You receive a screenshot after every action.\n"
    "Rules:\n"
    "1. Issue exactly one `computer` tool call per turn.\n"
    "2. Never type shell commands, never try to leave the sandbox application, "
    "never navigate to external websites.\n"
    "3. Prefer clicking on visible controls; type only into focused inputs.\n"
    "4. When the task is fully done, stop calling tools and reply with a short "
    "plain-text summary. If the task asks for a value, the summary must contain "
    "that exact value.\n"
    "5. If the screen did not change after an action, try a different approach."
)


class StubReasoner:
    """Deterministic reasoner driven by a scripted plan (offline baseline).

    Args:
        max_wait_retries: How many times to wait for a plan target to become
            visible before giving up on the task.
    """

    name = "stub"

    def __init__(self, max_wait_retries: int = 3) -> None:
        self.history: list[Action] = []
        self._plan: list[dict[str, Any]] = []
        self._plan_index = 0
        self._waits = 0
        self.max_wait_retries = max_wait_retries
        self.feedback_notes: list[str] = []

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Load the scripted plan for a task (if any) and clear history."""
        self.history = []
        self._plan_index = 0
        self._waits = 0
        self.feedback_notes = []
        self._plan = list((spec or {}).get("plan") or [])

    def feedback(self, note: str) -> None:
        """Store verifier feedback (the stub does not change its plan)."""
        self.feedback_notes.append(note)

    @staticmethod
    def _keyword_plan(task: str) -> list[dict[str, Any]]:
        t = task.lower()
        if "open" in t and ("file" in t or "folder" in t):
            return [
                {"action": "screenshot"},
                {"action": "click", "coords": [50, 50]},
                {"action": "type", "text": "My Documents"},
                {"action": "key", "key": "Return"},
                {"action": "task_complete", "text": "opened folder"},
            ]
        if "csv" in t or "extract" in t:
            return [
                {"action": "screenshot"},
                {"action": "click", "coords": [100, 200]},
                {"action": "key", "key": "ctrl+a"},
                {"action": "key", "key": "ctrl+c"},
                {"action": "task_complete", "text": "copied table"},
            ]
        return [
            {"action": "screenshot"},
            {"action": "click", "coords": [100, 100]},
            {"action": "task_complete", "text": "done"},
        ]

    def _resolve(self, step: dict[str, Any], observation: Observation) -> Action | None:
        """Turn a plan step into a concrete action, or ``None`` if the target is not visible."""
        kind = str(step["action"])
        if kind in {"click", "double_click", "right_click", "mouse_move"}:
            if "coords" in step:
                x, y = step["coords"]
                return Action(type=kind, coords=(int(x), int(y)))
            element = observation.find(str(step["target"]))
            if element is None:
                return None
            return Action(type=kind, coords=(element.x, element.y))
        if kind == "type":
            return Action(type="type", text=str(step["text"]))
        if kind == "key":
            return Action(type="key", key=str(step["key"]))
        if kind == "scroll":
            return Action(
                type="scroll",
                coords=tuple(step.get("coords", (512, 384))),
                scroll_direction=step.get("direction", "down"),
                scroll_amount=int(step.get("amount", 3)),
            )
        if kind == "wait":
            return Action(type="wait", duration_ms=int(step.get("duration_ms", 200)))
        if kind == "screenshot":
            return Action(type="screenshot")
        if kind == "task_complete":
            answer_from = step.get("answer_from")
            if answer_from:
                fields = [f.strip() for f in str(answer_from).split(",")]
                values = [observation.fields.get(f) for f in fields]
                if any(v is None for v in values):
                    return None
                return Action(type="task_complete", text=", ".join(str(v) for v in values))
            return Action(type="task_complete", text=str(step.get("text", "done")))
        raise ValueError(f"unknown plan action: {kind!r}")

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Emit the next planned action, waiting when its target is not yet visible."""
        plan = self._plan or self._keyword_plan(task)
        if self._plan_index >= len(plan):
            action = Action(type="task_complete", text="plan exhausted")
            self.history.append(action)
            return ReasonerOutput(action=action, reasoning=f"step {step}: plan exhausted")
        planned = plan[self._plan_index]
        resolved = self._resolve(planned, observation)
        if resolved is None:
            self._waits += 1
            if self._waits > self.max_wait_retries:
                action = Action(type="task_complete", text="gave up: target never appeared")
                reasoning = f"step {step}: target {planned.get('target') or planned.get('answer_from')!r} never became visible"
                self.history.append(action)
                return ReasonerOutput(action=action, reasoning=reasoning)
            action = Action(type="wait", duration_ms=200)
            reasoning = (
                f"step {step}: waiting for {planned.get('target') or planned.get('answer_from')!r} "
                f"to appear (attempt {self._waits}/{self.max_wait_retries})"
            )
            self.history.append(action)
            return ReasonerOutput(action=action, reasoning=reasoning)
        action = resolved
        self._waits = 0
        self._plan_index += 1
        self.history.append(action)
        target = planned.get("target") or planned.get("answer_from") or ""
        reasoning = f"step {step}: scripted {action.type}" + (f" on {target}" if target else "")
        return ReasonerOutput(action=action, reasoning=reasoning)


UNSUPPORTED_ACTIONS = {
    "left_click_drag",
    "left_mouse_down",
    "left_mouse_up",
    "hold_key",
    "triple_click",
    "cursor_position",
}


def parse_tool_input(tool_input: dict[str, Any]) -> tuple[Action, str | None]:
    """Map an Anthropic ``computer`` tool input to an :class:`Action`.

    Returns:
        The action and an optional error message to feed back to the model when
        the requested action is unsupported or malformed.
    """
    kind = str(tool_input.get("action", ""))
    coordinate = tool_input.get("coordinate")
    coords: tuple[int, int] | None = None
    if isinstance(coordinate, list | tuple) and len(coordinate) == 2:
        coords = (int(coordinate[0]), int(coordinate[1]))

    if kind in {"left_click", "middle_click"}:
        if coords is None:
            return Action(type="screenshot"), "click requires a coordinate"
        return Action(type="click", coords=coords), None
    if kind == "double_click":
        if coords is None:
            return Action(type="screenshot"), "double_click requires a coordinate"
        return Action(type="double_click", coords=coords), None
    if kind == "right_click":
        if coords is None:
            return Action(type="screenshot"), "right_click requires a coordinate"
        return Action(type="right_click", coords=coords), None
    if kind == "mouse_move":
        if coords is None:
            return Action(type="screenshot"), "mouse_move requires a coordinate"
        return Action(type="mouse_move", coords=coords), None
    if kind == "type":
        text = tool_input.get("text")
        if not isinstance(text, str) or not text:
            return Action(type="screenshot"), "type requires non-empty text"
        return Action(type="type", text=text), None
    if kind == "key":
        key = tool_input.get("text")
        if not isinstance(key, str) or not key:
            return Action(type="screenshot"), "key requires a key name"
        return Action(type="key", key=key), None
    if kind == "scroll":
        direction = str(tool_input.get("scroll_direction", "down"))
        if direction not in {"up", "down", "left", "right"}:
            direction = "down"
        amount = int(tool_input.get("scroll_amount", 3) or 3)
        return (
            Action(
                type="scroll",
                coords=coords,
                scroll_direction=direction,
                scroll_amount=max(0, min(amount, 100)),
            ),
            None,
        )
    if kind == "wait":
        seconds = float(tool_input.get("duration", 1) or 1)
        return Action(type="wait", duration_ms=int(min(seconds, 60) * 1000)), None
    if kind == "screenshot":
        return Action(type="screenshot"), None
    if kind in UNSUPPORTED_ACTIONS:
        return Action(type="screenshot"), f"action {kind!r} is not supported in this sandbox"
    return Action(type="screenshot"), f"unknown action {kind!r}"


class ClaudeComputerUseReasoner:
    """Anthropic Computer Use reasoner.

    Keeps the conversation (task, screenshots, tool calls) between steps and
    prunes old screenshots so the context stays bounded.

    Args:
        llm: Shared :class:`LLMClient` (pinned model, timeout, retries).
        tool_version: ``computer_20250124`` (default) or ``computer_20241022``.
        display: Screen size sent to the tool definition.
        max_images: Screenshots kept in context (older ones are replaced by text).
    """

    name = "claude"

    def __init__(
        self,
        llm: LLMClient,
        *,
        tool_version: str = "computer_20250124",
        display: tuple[int, int] = (1024, 768),
        max_images: int = 3,
    ) -> None:
        self.llm = llm
        self.tool_version = tool_version
        self.display = display
        self.max_images = max_images
        self.messages: list[dict[str, Any]] = []
        self._pending_tool_use_ids: list[str] = []
        self._pending_error: str | None = None
        self._feedback: str | None = None

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Start a fresh conversation."""
        self.messages = []
        self._pending_tool_use_ids = []
        self._pending_error = None
        self._feedback = None

    def feedback(self, note: str) -> None:
        """Attach a verifier note to the next tool result."""
        self._feedback = note

    def _prune_images(self) -> None:
        """Keep only the last ``max_images`` image blocks in the conversation."""
        seen = 0
        for message in reversed(self.messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                blocks = block.get("content") if block.get("type") == "tool_result" else [block]
                if not isinstance(blocks, list):
                    continue
                for inner in blocks:
                    if inner.get("type") != "image":
                        continue
                    seen += 1
                    if seen > self.max_images:
                        inner.clear()
                        inner.update(text_block("[earlier screenshot removed to save context]"))

    def _user_turn(self, task: str, observation: Observation, step: int) -> dict[str, Any]:
        png = observation.png or b""
        if not self._pending_tool_use_ids:
            content = [
                text_block(f"Task: {task}\nThis is the current screen (step {step})."),
                image_block(png),
            ]
            return {"role": "user", "content": content}
        results: list[dict[str, Any]] = []
        first, *rest = self._pending_tool_use_ids
        result_content: list[dict[str, Any]] = []
        notes = [n for n in (self._pending_error, self._feedback) if n]
        if notes:
            result_content.append(text_block("; ".join(notes)))
        result_content.append(image_block(png))
        results.append({"type": "tool_result", "tool_use_id": first, "content": result_content})
        for extra in rest:
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": extra,
                    "content": [text_block("skipped: only one action per turn is executed")],
                    "is_error": True,
                }
            )
        self._pending_error = None
        self._feedback = None
        return {"role": "user", "content": results}

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Call the model with the current screenshot and parse its tool call."""
        self.messages.append(self._user_turn(task, observation, step))
        self._prune_images()
        reply = await self.llm.computer_use(
            self.messages,
            system=SYSTEM_PROMPT.format(width=self.display[0], height=self.display[1]),
            tool_version=self.tool_version,
            display_width=self.display[0],
            display_height=self.display[1],
        )
        self.messages.append({"role": "assistant", "content": content_to_params(reply.raw_content)})
        reasoning = reply.text or "(no reasoning text)"
        if not reply.tool_uses:
            self._pending_tool_use_ids = []
            action = Action(type="task_complete", text=reply.text[:2000] or "done")
            logger.info(f"model ended turn ({reply.stop_reason}); marking task complete")
            return ReasonerOutput(
                action=action,
                reasoning=reasoning,
                tokens_in=reply.tokens_in,
                tokens_out=reply.tokens_out,
                cost_usd=reply.cost_usd,
            )
        self._pending_tool_use_ids = [str(t["id"]) for t in reply.tool_uses]
        action, error = parse_tool_input(reply.tool_uses[0]["input"])
        self._pending_error = error
        if error:
            reasoning = f"{reasoning} [tool error: {error}]"
        return ReasonerOutput(
            action=action,
            reasoning=reasoning,
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            cost_usd=reply.cost_usd,
        )
