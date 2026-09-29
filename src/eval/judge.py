"""Final validator (LLM-as-judge) with an explicit, numbered rubric.

The judge is only an *additional* signal: the deterministic checkers in
``src/sandbox/checkers.py`` decide success. The rubric lives in code so it is
versioned and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.llm import LLMClient, image_block, text_block

FINAL_VALIDATOR_RUBRIC: list[tuple[int, str, str]] = [
    (
        1,
        "Goal satisfied",
        "The final screen and/or the agent's answer show the task's goal was reached "
        "(e.g. the form shows a confirmation with the requested values). Example pass: "
        "'Contact form submitted: Alice / a@b.com'. Example fail: form still empty.",
    ),
    (
        2,
        "Exact values",
        "Every value requested by the task (names, emails, numbers, page names) appears "
        "exactly, without typos or extra characters. Example fail: 'a@b.co' instead of 'a@b.com'.",
    ),
    (
        3,
        "No collateral change",
        "Nothing outside the task was modified: no extra records created, no settings "
        "changed, no navigation away from the app.",
    ),
    (
        4,
        "No error state",
        "No error dialog, validation message or unexpected pop-up is visible on the final screen.",
    ),
    (
        5,
        "Honest answer",
        "The agent's final message does not claim success that the screen contradicts, and "
        "does not invent values that are not visible.",
    ),
]
"""Numbered criteria; each scores 0 or 1, the task passes with >= 4/5."""

PASS_THRESHOLD = 4


def rubric_text() -> str:
    """Render the rubric for the prompt."""
    lines = [f"{n}. {title}: {desc}" for n, title, desc in FINAL_VALIDATOR_RUBRIC]
    return "\n".join(lines)


JUDGE_PROMPT = (
    "You are the final validator of a computer-use agent.\n"
    "Task given to the agent: {task}\n"
    "Agent's final message: {answer}\n\n"
    "Score each criterion with 0 or 1:\n{rubric}\n\n"
    "Reply with one line per criterion in the form '<n>: <0|1> - <reason>' and a last "
    "line 'TOTAL: <sum>'."
)


@dataclass
class JudgeResult:
    """Outcome of the final validator."""

    scores: dict[int, int]
    total: int
    passed: bool
    rationale: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


def parse_judge_reply(text: str) -> tuple[dict[int, int], int]:
    """Parse ``'<n>: <0|1> - reason'`` lines into scores; the total is recomputed."""
    scores: dict[int, int] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\d)\s*[:.)]\s*([01])\b", line)
        if match:
            scores[int(match.group(1))] = int(match.group(2))
    total = sum(scores.get(n, 0) for n, _, _ in FINAL_VALIDATOR_RUBRIC)
    return scores, total


class ClaudeFinalValidator:
    """LLM judge over the final screenshot and answer (mock-tested)."""

    name = "claude-judge"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def validate(self, task: str, answer: str, final_png: bytes | None) -> JudgeResult:
        """Score the run against :data:`FINAL_VALIDATOR_RUBRIC`."""
        content: list[dict[str, Any]] = []
        if final_png is not None:
            content.append(image_block(final_png))
        content.append(
            text_block(JUDGE_PROMPT.format(task=task, answer=answer or "(none)", rubric=rubric_text()))
        )
        reply = await self._llm.complete(content, max_tokens=400)
        scores, total = parse_judge_reply(reply.text)
        return JudgeResult(
            scores=scores,
            total=total,
            passed=total >= PASS_THRESHOLD,
            rationale=reply.text[:1000],
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            cost_usd=reply.cost_usd,
        )
