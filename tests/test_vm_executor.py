from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.vm_executor import FakeVM, XdoToolVM, build_vm
from src.api.schemas import Action


async def test_fake_vm_records_events_and_fields():
    vm = FakeVM()
    await vm.reset({"app": "forms.html"})
    obs = await vm.screenshot()
    assert obs.url == "fake://forms.html"
    assert obs.find("#ok") is not None
    assert obs.fields["version"] == "0.0-fake"
    assert "Clicked" not in await vm.execute(Action(type="click", coords=(1, 2)))
    for action in [
        Action(type="type", text="x"),
        Action(type="key", key="Return"),
        Action(type="scroll"),
        Action(type="wait", duration_ms=1),
        Action(type="task_complete"),
        Action(type="screenshot"),
    ]:
        assert await vm.execute(action)
    assert vm.typed == ["x"]
    assert len(vm.events) == 7
    assert await vm.check({"type": "answer_contains", "value": "a"}, "A") is True
    assert await vm.check({"type": "hash_equals", "value": "#x"}, "A") is False
    await vm.close()


class FakeRunner:
    def __init__(self, rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.rc = rc

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        return SimpleNamespace(returncode=self.rc, stderr="", stdout="")


async def test_xdotool_vm_translates_actions(tmp_path):
    runner = FakeRunner()
    vm = XdoToolVM(screenshot_dir=str(tmp_path), runner=runner)
    obs = await vm.screenshot()
    assert runner.calls[0][0] == "scrot"
    assert obs.png is None  # scrot mocked: no file written
    await vm.execute(Action(type="click", coords=(10, 20)))
    await vm.execute(Action(type="double_click", coords=(10, 20)))
    await vm.execute(Action(type="right_click", coords=(10, 20)))
    await vm.execute(Action(type="mouse_move", coords=(1, 1)))
    await vm.execute(Action(type="type", text="hello"))
    await vm.execute(Action(type="key", key="enter"))
    await vm.execute(Action(type="scroll", scroll_direction="up", scroll_amount=2))
    assert await vm.execute(Action(type="wait", duration_ms=1)) == "Waited 1ms"
    assert await vm.execute(Action(type="task_complete")) == "Action: task_complete"
    cmds = runner.calls[1:]
    assert cmds[0] == ["xdotool", "mousemove", "10", "20", "click", "1"]
    assert cmds[1] == ["xdotool", "mousemove", "10", "20", "click", "--repeat", "2", "1"]
    assert cmds[2] == ["xdotool", "mousemove", "10", "20", "click", "3"]
    assert cmds[3] == ["xdotool", "mousemove", "1", "1"]
    assert cmds[4] == ["xdotool", "type", "--delay", "20", "hello"]
    assert cmds[5] == ["xdotool", "key", "Return"]
    assert cmds[6] == ["xdotool", "click", "--repeat", "2", "4"]
    assert await vm.check({"type": "answer_contains", "value": "x"}, "x") is True
    await vm.reset()
    await vm.close()


async def test_xdotool_vm_reports_failures(tmp_path):
    vm = XdoToolVM(screenshot_dir=str(tmp_path), runner=FakeRunner(rc=1))
    assert "failed" in await vm.execute(Action(type="key", key="a"))


def test_xdotool_requires_binaries(mocker):
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(RuntimeError):
        XdoToolVM()


def test_build_vm_modes(settings, mocker):
    assert isinstance(build_vm(settings), FakeVM)
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(RuntimeError):
        build_vm(settings.model_copy(update={"vm_mode": "xdotool"}))
    pw = build_vm(settings.model_copy(update={"vm_mode": "playwright"}))
    assert pw.name == "playwright"
