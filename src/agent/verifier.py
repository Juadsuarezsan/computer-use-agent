"""Per-step verifiers: did the last action have a visible effect?

* :class:`HeuristicVerifier` — offline, compares screen digests before/after.
* :class:`ClaudeVerifier` — asks the pinned model with both screenshots
  (tested with a mocked client; never called without ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.agent.llm import LLMClient, image_block, text_block
from src.agent.protocols import Observation, VerifierResult
from src.api.schemas import Action

NO_EFFECT_ACTIONS = {"screenshot", "wait", "mouse_move", "task_complete"}


class HeuristicVerifier:
    """Flags pointer/keyboard actions that changed nothing on screen."""

    name = "heuristic"

    async def verify(self, before: Observation, action: Action, after: Observation) -> VerifierResult:
        """Return ``ok=False`` when a state-changing action left the screen identical."""
        if action.type in NO_EFFECT_ACTIONS:
            return VerifierResult(ok=True, note="no-op action")
        if before.digest() == after.digest():
            return VerifierResult(ok=False, note="no visible change after action")
        return VerifierResult(ok=True, note="screen changed")


VERIFIER_PROMPT = (
    "You are verifying one step of a computer-use agent. The first image is the "
    "screen BEFORE the action, the second is AFTER. Action: {action}.\n"
    "Answer with exactly one line: 'OK: <what changed>' if the action had the "
    "intended effect, or 'FAIL: <why>' if nothing changed, an unexpected dialog "
    "appeared, or an error is visible."
)


class ClaudeVerifier:
    """LLM verifier using two screenshots (before/after)."""

    name = "claude"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def verify(self, before: Observation, action: Action, after: Observation) -> VerifierResult:
        """Ask the model whether the action succeeded; falls back to the heuristic on no image."""
        if before.png is None or after.png is None:
            return await HeuristicVerifier().verify(before, action, after)
        content: list[dict[str, Any]] = [
            image_block(before.png),
            image_block(after.png),
            text_block(VERIFIER_PROMPT.format(action=action.summary())),
        ]
        reply = await self._llm.complete(content, max_tokens=120)
        text = reply.text.strip()
        ok = text.upper().startswith("OK")
        logger.debug(f"claude verifier: {text[:120]}")
        return VerifierResult(
            ok=ok,
            note=text[:200],
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            cost_usd=reply.cost_usd,
        )
