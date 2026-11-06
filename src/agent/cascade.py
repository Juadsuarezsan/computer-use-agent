"""Cascade: the local model acts first; a fallback reasoner takes over only when it is stuck.

The switch happens after ``escalate_after`` consecutive verifier failures and
lasts until the end of the task. Without a fallback (the default: no paid API
is called unless the operator configures one) the cascade is just the primary
reasoner, and the number of escalations stays visible in the trace.
"""

from __future__ import annotations

from typing import Any

from src.agent.protocols import Observation, Reasoner, ReasonerOutput


class CascadeReasoner:
    """Primary reasoner with an optional fallback after repeated verifier failures.

    Args:
        primary: Cheap policy used first (the local model).
        fallback: Expensive policy (Claude) or ``None`` to never escalate.
        escalate_after: Consecutive verifier failures that trigger the switch.
    """

    def __init__(
        self, primary: Reasoner, fallback: Reasoner | None = None, *, escalate_after: int = 2
    ) -> None:
        if escalate_after < 1:
            raise ValueError("escalate_after must be >= 1")
        self.primary = primary
        self.fallback = fallback
        self.escalate_after = escalate_after
        self.name = f"cascade({primary.name}->{fallback.name if fallback else 'none'})"
        self._spec: dict[str, Any] | None = None
        self._consecutive = 0
        self._failed_since_last = False
        self.escalated = False
        self.escalations = 0

    @property
    def active(self) -> Reasoner:
        """The reasoner currently deciding."""
        if self.escalated and self.fallback is not None:
            return self.fallback
        return self.primary

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Start a task on the primary reasoner."""
        self._spec = spec
        self._consecutive = 0
        self._failed_since_last = False
        self.escalated = False
        self.primary.reset(spec)
        if self.fallback is not None:
            self.fallback.reset(spec)

    def feedback(self, note: str) -> None:
        """Count verifier failures and forward the note to the active reasoner."""
        if note.startswith("verifier:"):
            self._failed_since_last = True
        self.active.feedback(note)

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Escalate if the primary is stuck, then delegate."""
        self._consecutive = self._consecutive + 1 if self._failed_since_last else 0
        self._failed_since_last = False
        if not self.escalated and self.fallback is not None and self._consecutive >= self.escalate_after:
            self.escalated = True
            self.escalations += 1
            self.fallback.reset(self._spec)
        out = await self.active.next_action(task, observation, step)
        out.reasoning = f"[{self.active.name}] {out.reasoning}"
        return out
