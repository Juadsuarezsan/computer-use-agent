"""ClaudeComputerUseReasoner + LLMClient with a mocked Anthropic client (never the real API)."""

from __future__ import annotations

from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from src.agent.llm import BETA_FLAGS, LLMClient, content_to_params, image_block
from src.agent.protocols import Observation
from src.agent.reasoner import ClaudeComputerUseReasoner, parse_tool_input
from tests.conftest import make_message

PNG = b"\x89PNG\r\n\x1a\nfake"


def _obs(step: int = 1) -> Observation:
    return Observation(screenshot_path=f"s{step}.png", description="screen", png=PNG, url="file:///x")


def test_llm_client_requires_key_or_client():
    with pytest.raises(ValueError):
        LLMClient(model="claude-sonnet-4-5-20250929", api_key=None)


def test_llm_client_builds_real_sdk_client_with_timeout():
    client = LLMClient(model="claude-sonnet-4-5-20250929", api_key="sk-test", timeout_s=12.5)
    assert isinstance(client._client, anthropic.AsyncAnthropic)
    assert client._client.timeout == 12.5
    assert client._client.max_retries == 0  # tenacity owns retries


async def test_tool_use_click_is_parsed_into_action(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.return_value = make_message(
        text="I will click the Contact tab.", tool_input={"action": "left_click", "coordinate": [200, 30]}
    )
    reasoner = ClaudeComputerUseReasoner(llm, tool_version="computer_20250124", display=(1024, 768))
    reasoner.reset()
    out = await reasoner.next_action("Open the contact form", _obs(), 1)
    assert out.action.type == "click"
    assert out.action.coords == (200, 30)
    assert "Contact tab" in out.reasoning
    assert out.tokens_in == 1200 and out.tokens_out == 40
    assert out.cost_usd == pytest.approx((1200 * 3 + 40 * 15) / 1_000_000)

    kwargs = mock_anthropic.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["betas"] == [BETA_FLAGS["computer_20250124"]]
    tool = kwargs["tools"][0]
    assert tool["type"] == "computer_20250124"
    assert tool["name"] == "computer"
    assert tool["display_width_px"] == 1024 and tool["display_height_px"] == 768
    first_user = kwargs["messages"][0]
    assert first_user["role"] == "user"
    assert first_user["content"][1]["type"] == "image"
    assert first_user["content"][1]["source"]["media_type"] == "image/png"


async def test_second_turn_sends_tool_result_with_screenshot(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.side_effect = [
        make_message(tool_input={"action": "left_click", "coordinate": [1, 1]}),
        make_message(tool_input={"action": "type", "text": "Alice"}),
    ]
    reasoner = ClaudeComputerUseReasoner(llm)
    reasoner.reset()
    await reasoner.next_action("t", _obs(1), 1)
    reasoner.feedback("verifier: previous action failed")
    out = await reasoner.next_action("t", _obs(2), 2)
    assert out.action.type == "type" and out.action.text == "Alice"
    messages = mock_anthropic.beta.messages.create.call_args.kwargs["messages"]
    assert [m["role"] for m in messages[:3]] == ["user", "assistant", "user"]
    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_00"
    assert tool_result["content"][0]["type"] == "text"  # feedback note
    assert "verifier" in tool_result["content"][0]["text"]
    assert tool_result["content"][1]["type"] == "image"


async def test_end_turn_without_tool_use_means_task_complete(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.return_value = make_message(text="Done: the total is 48,250.00")
    reasoner = ClaudeComputerUseReasoner(llm)
    out = await reasoner.next_action("t", _obs(), 1)
    assert out.action.type == "task_complete"
    assert "48,250.00" in (out.action.text or "")


async def test_unsupported_action_becomes_screenshot_with_error(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.side_effect = [
        make_message(tool_input={"action": "left_click_drag", "coordinate": [1, 1]}),
        make_message(text="ok"),
    ]
    reasoner = ClaudeComputerUseReasoner(llm)
    out = await reasoner.next_action("t", _obs(), 1)
    assert out.action.type == "screenshot"
    assert "not supported" in out.reasoning
    await reasoner.next_action("t", _obs(2), 2)
    tool_result = mock_anthropic.beta.messages.create.call_args.kwargs["messages"][2]["content"][0]
    assert "not supported" in tool_result["content"][0]["text"]


async def test_parallel_tool_uses_only_first_executed(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.side_effect = [
        make_message(
            tool_inputs=[
                {"action": "left_click", "coordinate": [1, 1]},
                {"action": "type", "text": "x"},
            ]
        ),
        make_message(text="done"),
    ]
    reasoner = ClaudeComputerUseReasoner(llm)
    out = await reasoner.next_action("t", _obs(), 1)
    assert out.action.type == "click"
    await reasoner.next_action("t", _obs(2), 2)
    results = mock_anthropic.beta.messages.create.call_args.kwargs["messages"][2]["content"]
    assert len(results) == 2
    assert results[1]["is_error"] is True


async def test_old_screenshots_are_pruned(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.beta.messages.create.return_value = make_message(tool_input={"action": "screenshot"})
    reasoner = ClaudeComputerUseReasoner(llm, max_images=2)
    for step in range(1, 6):
        await reasoner.next_action("t", _obs(step), step)
    images = 0
    for message in reasoner.messages:
        for block in message["content"]:
            inner = block.get("content", [block]) if block.get("type") == "tool_result" else [block]
            images += sum(1 for b in inner if b.get("type") == "image")
    assert images == 2


async def test_retries_on_rate_limit_then_succeeds(llm: LLMClient, mock_anthropic: MagicMock, mocker):
    mocker.patch("tenacity.nap.time.sleep")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    mock_anthropic.beta.messages.create.side_effect = [
        anthropic.RateLimitError("slow down", response=response, body=None),
        anthropic.APIConnectionError(request=request),
        make_message(tool_input={"action": "left_click", "coordinate": [3, 4]}),
    ]
    reasoner = ClaudeComputerUseReasoner(llm)
    out = await reasoner.next_action("t", _obs(), 1)
    assert out.action.coords == (3, 4)
    assert mock_anthropic.beta.messages.create.await_count == 3
    assert llm.calls == 1


async def test_gives_up_after_max_retries(llm: LLMClient, mock_anthropic: MagicMock, mocker):
    mocker.patch("tenacity.nap.time.sleep")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    mock_anthropic.beta.messages.create.side_effect = anthropic.InternalServerError(
        "boom", response=response, body=None
    )
    reasoner = ClaudeComputerUseReasoner(llm)
    with pytest.raises(anthropic.InternalServerError):
        await reasoner.next_action("t", _obs(), 1)
    assert mock_anthropic.beta.messages.create.await_count == 3


async def test_non_retryable_error_is_not_retried(llm: LLMClient, mock_anthropic: MagicMock):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    mock_anthropic.beta.messages.create.side_effect = anthropic.BadRequestError(
        "bad", response=response, body=None
    )
    reasoner = ClaudeComputerUseReasoner(llm)
    with pytest.raises(anthropic.BadRequestError):
        await reasoner.next_action("t", _obs(), 1)
    assert mock_anthropic.beta.messages.create.await_count == 1


async def test_unknown_tool_version_raises(llm: LLMClient):
    with pytest.raises(ValueError):
        await llm.computer_use(
            [], system="s", tool_version="computer_1999", display_width=1, display_height=1
        )


async def test_complete_uses_plain_messages_api(llm: LLMClient, mock_anthropic: MagicMock):
    mock_anthropic.messages.create.return_value = make_message(text="OK: form submitted")
    reply = await llm.complete(
        [image_block(PNG), {"type": "text", "text": "verify"}], system="sys", max_tokens=50
    )
    assert reply.text == "OK: form submitted"
    kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys" and kwargs["max_tokens"] == 50
    assert llm.total_cost_usd > 0


@pytest.mark.parametrize(
    ("tool_input", "expected_type", "has_error"),
    [
        ({"action": "left_click", "coordinate": [10, 20]}, "click", False),
        ({"action": "middle_click", "coordinate": [10, 20]}, "click", False),
        ({"action": "left_click"}, "screenshot", True),
        ({"action": "double_click", "coordinate": [1, 2]}, "double_click", False),
        ({"action": "double_click"}, "screenshot", True),
        ({"action": "right_click", "coordinate": [1, 2]}, "right_click", False),
        ({"action": "right_click"}, "screenshot", True),
        ({"action": "mouse_move", "coordinate": [1, 2]}, "mouse_move", False),
        ({"action": "mouse_move"}, "screenshot", True),
        ({"action": "type", "text": "abc"}, "type", False),
        ({"action": "type", "text": ""}, "screenshot", True),
        ({"action": "key", "text": "ctrl+a"}, "key", False),
        ({"action": "key"}, "screenshot", True),
        (
            {"action": "scroll", "coordinate": [5, 5], "scroll_direction": "up", "scroll_amount": 2},
            "scroll",
            False,
        ),
        ({"action": "scroll", "scroll_direction": "sideways"}, "scroll", False),
        ({"action": "wait", "duration": 2}, "wait", False),
        ({"action": "screenshot"}, "screenshot", False),
        ({"action": "hold_key", "text": "shift"}, "screenshot", True),
        ({"action": "teleport"}, "screenshot", True),
    ],
)
def test_parse_tool_input(tool_input, expected_type, has_error):
    action, error = parse_tool_input(tool_input)
    assert action.type == expected_type
    assert (error is not None) is has_error


def test_parse_wait_caps_duration():
    action, _ = parse_tool_input({"action": "wait", "duration": 999})
    assert action.duration_ms == 60_000


def test_content_to_params_handles_dicts_and_models():
    msg = make_message(text="hi", tool_input={"action": "screenshot"})
    params = content_to_params(list(msg.content) + [{"type": "text", "text": "raw"}])
    assert [p["type"] for p in params] == ["text", "tool_use", "text"]
