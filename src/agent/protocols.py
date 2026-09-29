"""Interfaces shared by the agent loop: observation, VM, reasoner and verifier."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.api.schemas import Action


@dataclass(frozen=True)
class UIElement:
    """A visible interactive element on screen (from the DOM or accessibility tree).

    Attributes:
        selector: Stable CSS selector (``#id`` preferred) usable by scripted plans.
        role: ``button``, ``link``, ``textbox``, ``checkbox``, ``radio``, ``select`` ...
        label: Human-readable label (text, aria-label, placeholder or name).
        x: Center x (pixels, viewport coordinates).
        y: Center y (pixels, viewport coordinates).
    """

    selector: str
    role: str
    label: str
    x: int
    y: int


@dataclass
class Observation:
    """What the agent sees after one screenshot.

    Attributes:
        screenshot_path: Where the PNG was written (``""`` for in-memory VMs).
        description: Textual description of the screen (title, url, element list).
        png: Raw PNG bytes (``None`` for the fake VM).
        elements: Interactive elements with coordinates (empty for non-DOM VMs).
        fields: Named text values marked with ``data-field`` in fixture apps.
        url: Current URL or window title.
    """

    screenshot_path: str = ""
    description: str = ""
    png: bytes | None = None
    elements: list[UIElement] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    url: str = ""

    def digest(self) -> str:
        """Return a short stable hash of the visible state (screenshot + fields)."""
        h = hashlib.sha256()
        h.update(self.png or self.description.encode("utf-8"))
        h.update(repr(sorted(self.fields.items())).encode("utf-8"))
        h.update(self.url.encode("utf-8"))
        return h.hexdigest()[:16]

    def find(self, selector: str) -> UIElement | None:
        """Return the element with the given selector, if visible."""
        for el in self.elements:
            if el.selector == selector:
                return el
        return None


@dataclass
class ReasonerOutput:
    """Reasoner decision plus token accounting for the call."""

    action: Action
    reasoning: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class VerifierResult:
    """Outcome of the per-step verifier."""

    ok: bool
    note: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@runtime_checkable
class VM(Protocol):
    """Driver for a (virtual) desktop or browser."""

    name: str

    async def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Bring the environment to the task's start state."""

    async def screenshot(self) -> Observation:
        """Capture the current screen."""

    async def execute(self, action: Action) -> str:
        """Perform an action and return a human-readable result."""

    async def check(self, checker: dict[str, Any], answer: str | None) -> bool:
        """Evaluate a deterministic ground-truth checker against the live state."""

    async def close(self) -> None:
        """Release resources."""


@runtime_checkable
class Reasoner(Protocol):
    """Chooses the next action from the task and the current observation."""

    name: str

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Forget conversation history before a new task."""

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Decide the next action."""

    def feedback(self, note: str) -> None:
        """Receive a verifier note about the previous action (may be ignored)."""


@runtime_checkable
class Verifier(Protocol):
    """Judges whether the previous action had the expected effect."""

    name: str

    async def verify(self, before: Observation, action: Action, after: Observation) -> VerifierResult:
        """Compare the screens before/after ``action``."""
