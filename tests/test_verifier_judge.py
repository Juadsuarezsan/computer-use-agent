from unittest.mock import MagicMock

from src.agent.llm import LLMClient
from src.agent.protocols import Observation
from src.agent.verifier import ClaudeVerifier, HeuristicVerifier
from src.api.schemas import Action
from src.eval.judge import (
    FINAL_VALIDATOR_RUBRIC,
    PASS_THRESHOLD,
    ClaudeFinalValidator,
    parse_judge_reply,
    rubric_text,
)
from tests.conftest import make_message


async def test_heuristic_verifier_flags_no_change():
    before = Observation(description="same", url="a")
    after = Observation(description="same", url="a")
    result = await HeuristicVerifier().verify(before, Action(type="click", coords=(1, 1)), after)
    assert result.ok is False
    assert "no visible change" in result.note


async def test_heuristic_verifier_accepts_change_and_noops():
    before = Observation(description="a", url="a")
    after = Observation(description="b", url="a")
    assert (await HeuristicVerifier().verify(before, Action(type="click", coords=(1, 1)), after)).ok
    assert (await HeuristicVerifier().verify(before, Action(type="screenshot"), before)).ok


async def test_claude_verifier_parses_ok_and_fail(llm: LLMClient, mock_anthropic: MagicMock):
    verifier = ClaudeVerifier(llm)
    before = Observation(description="a", png=b"1")
    after = Observation(description="b", png=b"2")
    mock_anthropic.messages.create.return_value = make_message(text="OK: the form was submitted")
    ok = await verifier.verify(before, Action(type="click", coords=(1, 1)), after)
    assert ok.ok is True and ok.tokens_in == 1200
    mock_anthropic.messages.create.return_value = make_message(text="FAIL: an error dialog appeared")
    fail = await verifier.verify(before, Action(type="click", coords=(1, 1)), after)
    assert fail.ok is False and "error dialog" in fail.note
    content = mock_anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]


async def test_claude_verifier_falls_back_without_images(llm: LLMClient, mock_anthropic: MagicMock):
    result = await ClaudeVerifier(llm).verify(
        Observation(description="a"), Action(type="wait"), Observation()
    )
    assert result.ok is True
    mock_anthropic.messages.create.assert_not_called()


def test_rubric_is_numbered_and_rendered():
    assert [n for n, _, _ in FINAL_VALIDATOR_RUBRIC] == [1, 2, 3, 4, 5]
    text = rubric_text()
    assert text.startswith("1. Goal satisfied")
    assert "5. Honest answer" in text


def test_parse_judge_reply_recomputes_total():
    scores, total = parse_judge_reply("1: 1 - good\n2: 0 - typo\n3) 1 - ok\n4: 1\n5: 1 - fine\nTOTAL: 99")
    assert scores == {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
    assert total == 4


async def test_final_validator_pass_and_fail(llm: LLMClient, mock_anthropic: MagicMock):
    judge = ClaudeFinalValidator(llm)
    mock_anthropic.messages.create.return_value = make_message(text="1: 1\n2: 1\n3: 1\n4: 1\n5: 0")
    result = await judge.validate("submit form", "done", b"png")
    assert result.passed is True and result.total == PASS_THRESHOLD
    mock_anthropic.messages.create.return_value = make_message(text="1: 0\n2: 0\n3: 1\n4: 1\n5: 1")
    result = await judge.validate("submit form", "", None)
    assert result.passed is False and result.total == 3
    content = mock_anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "text"  # no image when final_png is None
    assert "1. Goal satisfied" in content[0]["text"]
