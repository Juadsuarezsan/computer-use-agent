"""Real headless-Chromium sandbox tests (skipped when the browser binary is missing)."""

from __future__ import annotations

import pytest

from src.agent.orchestrator import run_task
from src.agent.playwright_vm import PlaywrightVM, xdotool_key_to_playwright
from src.agent.reasoner import StubReasoner
from src.agent.verifier import HeuristicVerifier
from src.api.schemas import Action
from src.eval.tasks import get_task
from tests.conftest import REPO, requires_sandbox

pytestmark = [requires_sandbox, pytest.mark.sandbox]


@pytest.fixture()
async def vm(settings, tmp_path):
    vm = PlaywrightVM(
        chromium_path=settings.chromium_path,
        webapps_dir=str(REPO / "sandbox" / "webapps"),
        screenshot_dir=str(tmp_path),
    )
    yield vm
    await vm.close()


def test_key_translation():
    assert xdotool_key_to_playwright("ctrl+a") == "Control+a"
    assert xdotool_key_to_playwright("Return") == "Enter"
    assert xdotool_key_to_playwright("shift+Tab") == "Shift+Tab"
    assert xdotool_key_to_playwright("super+l") == "Meta+l"
    assert xdotool_key_to_playwright("F5") == "F5"
    assert xdotool_key_to_playwright("Page_Down") == "PageDown"


async def test_screenshot_and_element_map(vm: PlaywrightVM):
    await vm.reset({"id": "t", "app": "forms.html", "start": "#contact"})
    obs = await vm.screenshot()
    assert obs.png and obs.png[:4] == b"\x89PNG"
    assert obs.screenshot_path.endswith("t_001.png")
    assert obs.find("#contact-name") is not None
    assert obs.find("#reg-username") is None  # hidden view is not listed
    assert "interactive elements" in obs.description
    assert obs.url.endswith("forms.html#contact")


async def test_execute_click_type_key_scroll_and_check(vm: PlaywrightVM):
    await vm.reset({"id": "t", "app": "forms.html", "start": "#contact"})
    obs = await vm.screenshot()
    name = obs.find("#contact-name")
    assert name is not None
    assert "click" in await vm.execute(Action(type="click", coords=(name.x, name.y)))
    await vm.execute(Action(type="type", text="Bob"))
    await vm.execute(Action(type="key", key="ctrl+a"))
    await vm.execute(Action(type="type", text="Alice"))
    email = obs.find("#contact-email")
    await vm.execute(Action(type="click", coords=(email.x, email.y)))
    await vm.execute(Action(type="type", text="a@b.com"))
    await vm.execute(Action(type="key", key="Return"))
    assert await vm.check(
        {"type": "dom_attr_equals", "selector": "#contact-result", "attr": "data-name", "value": "Alice"},
        None,
    )
    assert not await vm.check(
        {"type": "dom_attr_equals", "selector": "#contact-result", "attr": "data-name", "value": "Bob"}, None
    )
    assert "scrolled" in await vm.execute(Action(type="scroll", coords=(500, 400)))
    assert "moved" in await vm.execute(Action(type="mouse_move", coords=(5, 5)))
    assert "waited" in await vm.execute(Action(type="wait", duration_ms=5))
    assert "double_click" in await vm.execute(Action(type="double_click", coords=(5, 5)))
    assert "right_click" in await vm.execute(Action(type="right_click", coords=(5, 5)))
    assert await vm.execute(Action(type="task_complete")) == "task marked complete"
    assert "no-op" in await vm.execute(Action(type="screenshot"))


async def test_missing_app_raises(vm: PlaywrightVM):
    with pytest.raises(FileNotFoundError):
        vm.app_url("nope.html")


async def test_full_task_with_stub_and_checker(vm: PlaywrightVM):
    spec = get_task("ff-01")
    assert spec is not None
    out = await run_task(
        task=spec.task, task_spec=spec.as_spec(), vm=vm, reasoner=StubReasoner(), verifier=HeuristicVerifier()
    )
    assert out.success is True and out.ground_truth_checked is True
    assert out.total_steps == len(spec.plan)
    assert all(s.screenshot_path for s in out.steps)


async def test_wrong_plan_fails_checker(vm: PlaywrightVM):
    spec = get_task("ff-01").as_spec()
    spec["plan"] = [
        s if s.get("text") != "Alice" else {"action": "type", "text": "Mallory"} for s in spec["plan"]
    ]
    out = await run_task(task=spec["task"], task_spec=spec, vm=vm, reasoner=StubReasoner())
    assert out.success is False and out.ground_truth_checked is True


async def test_dashboard_login_and_breadcrumb(vm: PlaywrightVM):
    spec = get_task("wn-04")
    out = await run_task(task=spec.task, task_spec=spec.as_spec(), vm=vm, reasoner=StubReasoner())
    assert out.success is True
    # sidebar Home also lands on #/home but must NOT satisfy the breadcrumb checker
    alt = spec.as_spec()
    alt["plan"] = [{"action": "click", "target": "#side-home"}, {"action": "task_complete", "text": "x"}]
    out = await run_task(task=spec.task, task_spec=alt, vm=vm, reasoner=StubReasoner())
    assert out.success is False
