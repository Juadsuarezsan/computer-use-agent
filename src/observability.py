"""Structured logging, trace ids, token/cost accounting and LangSmith wiring.

Every task run gets a ``trace_id``; every graph node logs its input and output
state through :func:`log_node`. Token counts and USD cost are computed from the
pinned pricing table so the eval harness can report cost per task.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from loguru import logger

from src.config import Settings

# USD per million tokens (input, output). Only models we pin are listed; an
# unknown model yields cost 0 and a warning so we never invent numbers silently.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def new_trace_id() -> str:
    """Return a new opaque trace id."""
    return uuid.uuid4().hex


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return the USD cost of a call from the pricing table.

    Args:
        model: Pinned model id.
        tokens_in: Input tokens (including image tokens as billed by the API).
        tokens_out: Output tokens.

    Returns:
        Cost in USD (``0.0`` and a warning when the model is not in the table).
    """
    pricing = MODEL_PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        logger.warning(f"no pricing for model {model!r}; cost reported as 0")
        return 0.0
    price_in, price_out = pricing
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a compact structured format (idempotent)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:HH:mm:ss.SSS} | {level:<7} | {extra[trace_id]} | {name}:{function} | {message}",
    )
    logger.configure(extra={"trace_id": "-"})


def log_node(trace_id: str, node: str, phase: str, state: dict[str, Any]) -> None:
    """Log a compact summary of a graph node's input or output state.

    Args:
        trace_id: Task trace id.
        node: Graph node name.
        phase: ``"in"`` or ``"out"``.
        state: The (partial) LangGraph state to summarise.
    """
    summary: dict[str, Any] = {}
    for key, value in state.items():
        if key in {"observation", "prev_observation"}:
            summary[key] = getattr(value, "url", "") if value is not None else None
        elif key == "steps_taken":
            summary[key] = len(value) if isinstance(value, list) else value
        elif key == "action":
            summary[key] = value.summary() if value is not None else None
        elif isinstance(value, int | float | bool | str) or value is None:
            summary[key] = value if not isinstance(value, str) else value[:80]
    logger.bind(trace_id=trace_id).info(f"node={node} phase={phase} {summary}")


def configure_langsmith(settings: Settings) -> bool:
    """Export LangSmith environment variables so LangGraph traces every run.

    LangGraph/LangChain read ``LANGCHAIN_TRACING_V2``, ``LANGCHAIN_API_KEY`` and
    ``LANGCHAIN_PROJECT`` from the process environment; we only mirror the
    validated settings there. Nothing is sent when tracing is disabled or no
    key is configured.

    Returns:
        ``True`` when tracing is active.
    """
    if not (settings.langchain_tracing_v2 and settings.langchain_api_key):
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info(f"LangSmith tracing enabled for project {settings.langchain_project!r}")
    return True
