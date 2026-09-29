"""Shared fixtures: fixed seeds, offline settings, fake VM, mocked Anthropic client."""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

SEED = 20260516
REPO = Path(__file__).resolve().parent.parent
CHROMIUM = Path(os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium"))

# Offline settings for the whole test session (never hits the network).
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["VM_MODE"] = "fake"
os.environ["AUDIT_SQLITE_PATH"] = ":memory:"
os.environ["DATABASE_URL"] = ""
os.environ["CORS_ORIGINS"] = "http://testserver"
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["SCREENSHOT_DIR"] = str(REPO / "data" / "screenshots" / "tests")


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)


@pytest.fixture()
def settings() -> Any:
    from src.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture()
def fake_vm() -> Any:
    from src.agent.vm_executor import FakeVM

    return FakeVM()


@pytest.fixture()
def stub_reasoner() -> Any:
    from src.agent.reasoner import StubReasoner

    return StubReasoner()


@pytest.fixture()
def tasks() -> Any:
    from src.eval.tasks import load_tasks

    return load_tasks(REPO / "data" / "eval" / "tasks.json")


def make_message(
    *,
    text: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_inputs: list[dict[str, Any]] | None = None,
    stop_reason: str = "tool_use",
    tokens: tuple[int, int] = (1200, 40),
) -> SimpleNamespace:
    """Build an object shaped like ``anthropic.types.beta.BetaMessage``."""
    content: list[Any] = []
    if text is not None:
        content.append(
            SimpleNamespace(type="text", text=text, model_dump=lambda **_: {"type": "text", "text": text})
        )
    inputs = tool_inputs if tool_inputs is not None else ([tool_input] if tool_input is not None else [])
    for i, inp in enumerate(inputs):
        tool_id = f"toolu_{i:02d}"
        content.append(
            SimpleNamespace(
                type="tool_use",
                id=tool_id,
                name="computer",
                input=inp,
                model_dump=lambda inp=inp, tool_id=tool_id, **_: {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "computer",
                    "input": inp,
                },
            )
        )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason if inputs else "end_turn",
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


@pytest.fixture()
def mock_anthropic() -> MagicMock:
    """A stand-in for ``anthropic.AsyncAnthropic`` with async ``create`` methods."""
    client = MagicMock()
    client.beta.messages.create = AsyncMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture()
def llm(mock_anthropic: MagicMock) -> Any:
    from src.agent.llm import LLMClient

    return LLMClient(model="claude-sonnet-4-5-20250929", api_key=None, client=mock_anthropic, max_retries=3)


@pytest.fixture()
def api_client() -> Any:
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        yield client


def sandbox_available() -> bool:
    return CHROMIUM.exists()


requires_sandbox = pytest.mark.skipif(not sandbox_available(), reason="Chromium sandbox binary not available")
