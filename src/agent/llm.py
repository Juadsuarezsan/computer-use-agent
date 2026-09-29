"""Thin wrapper around the Anthropic SDK with timeout, retries and cost accounting.

All LLM traffic of the project (reasoner, verifier, final validator) goes
through :class:`LLMClient` so timeouts, ``tenacity`` retries and token/cost
bookkeeping are defined once. The client is never constructed without an API
key; tests inject a mocked ``AsyncAnthropic``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, cast

import anthropic
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.observability import estimate_cost_usd

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

BETA_FLAGS: dict[str, str] = {
    "computer_20250124": "computer-use-2025-01-24",
    "computer_20241022": "computer-use-2024-10-22",
}


def image_block(png: bytes) -> dict[str, Any]:
    """Return a base64 ``image`` content block for a PNG screenshot."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png).decode("ascii"),
        },
    }


def text_block(text: str) -> dict[str, Any]:
    """Return a ``text`` content block."""
    return {"type": "text", "text": text}


@dataclass
class LLMReply:
    """Normalised response of one Messages API call."""

    text: str
    tool_uses: list[dict[str, Any]]
    raw_content: list[Any]
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


class LLMClient:
    """Messages API client with pinned model, timeout, retries and cost tracking.

    Args:
        model: Pinned model id (e.g. ``claude-sonnet-4-5-20250929``).
        api_key: Anthropic API key.
        timeout_s: Per-request timeout in seconds.
        max_retries: Total attempts per call (exponential backoff between them).
        max_tokens: Default output budget.
        client: Optional pre-built ``AsyncAnthropic`` (tests pass a mock).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 1024,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("LLMClient requires an api_key or an injected client")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # SDK-level retries are disabled: tenacity owns the retry policy.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0
        self.calls = 0

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )

    def _account(self, message: Any) -> tuple[int, int, float]:
        usage = getattr(message, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        cost = estimate_cost_usd(self.model, tokens_in, tokens_out)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd = round(self.total_cost_usd + cost, 6)
        self.calls += 1
        return tokens_in, tokens_out, cost

    @staticmethod
    def _normalise(message: Any, tokens: tuple[int, int, float]) -> LLMReply:
        texts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                texts.append(str(block.text))
            elif block_type == "tool_use":
                tool_uses.append({"id": str(block.id), "name": str(block.name), "input": dict(block.input)})
        tokens_in, tokens_out, cost = tokens
        return LLMReply(
            text="\n".join(texts).strip(),
            tool_uses=tool_uses,
            raw_content=list(message.content),
            stop_reason=str(getattr(message, "stop_reason", "") or ""),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

    async def complete(
        self,
        content: list[dict[str, Any]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMReply:
        """Single-turn call without tools (verifier, final validator).

        Args:
            content: User content blocks (text/image).
            system: Optional system prompt.
            max_tokens: Output budget override.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system
        try:
            async for attempt in self._retrying():
                with attempt:
                    message = await self._client.messages.create(**kwargs)
        except RetryError as exc:  # pragma: no cover - reraise=True makes this unreachable
            raise exc.last_attempt.exception() or exc from exc
        return self._normalise(message, self._account(message))

    async def computer_use(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tool_version: str,
        display_width: int,
        display_height: int,
        max_tokens: int | None = None,
    ) -> LLMReply:
        """Multi-turn call with the Anthropic ``computer`` tool (beta Messages API).

        Args:
            messages: Full conversation (user/assistant turns, tool results).
            system: System prompt.
            tool_version: ``computer_20250124`` or ``computer_20241022``.
            display_width: Screen width in pixels.
            display_height: Screen height in pixels.
            max_tokens: Output budget override.

        Raises:
            ValueError: for an unknown tool version.
        """
        beta = BETA_FLAGS.get(tool_version)
        if beta is None:
            raise ValueError(f"unsupported computer-use tool version: {tool_version!r}")
        tool: dict[str, Any] = {
            "type": tool_version,
            "name": "computer",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": 1,
        }
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "tools": [tool],
            "messages": messages,
            "betas": [beta],
        }
        try:
            async for attempt in self._retrying():
                with attempt:
                    message = await self._client.beta.messages.create(**kwargs)
        except RetryError as exc:  # pragma: no cover - reraise=True makes this unreachable
            raise exc.last_attempt.exception() or exc from exc
        reply = self._normalise(message, self._account(message))
        logger.debug(
            f"computer_use call: stop={reply.stop_reason} tools={len(reply.tool_uses)} "
            f"in={reply.tokens_in} out={reply.tokens_out}"
        )
        return reply


def content_to_params(raw_content: list[Any]) -> list[dict[str, Any]]:
    """Convert SDK response blocks into request-compatible dicts."""
    out: list[dict[str, Any]] = []
    for block in raw_content:
        if hasattr(block, "model_dump"):
            out.append(cast(dict[str, Any], block.model_dump(exclude_none=True)))
        elif isinstance(block, dict):
            out.append(dict(block))
    return out
