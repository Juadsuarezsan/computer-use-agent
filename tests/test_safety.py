from src.api.schemas import Action
from src.safety.blocklist import check_action


def test_blocks_rm_rf():
    a = Action(type="type", text="rm -rf / --no-preserve-root")
    v = check_action(a)
    assert v.allowed is False
    assert "rm -rf" in v.reason


def test_blocks_sudo():
    a = Action(type="type", text="sudo apt-get install evil")
    v = check_action(a)
    assert v.allowed is False


def test_blocks_dangerous_key_combo():
    a = Action(type="key", key="ctrl+alt+del")
    v = check_action(a)
    assert v.allowed is False


def test_allows_normal_typing():
    a = Action(type="type", text="Hello world, this is a normal message.")
    assert check_action(a).allowed is True


def test_allows_safe_click():
    a = Action(type="click", coords=(100, 200))
    assert check_action(a).allowed is True


def test_blocks_click_without_coords():
    a = Action(type="click")
    v = check_action(a)
    assert v.allowed is False


def test_allows_normal_key():
    a = Action(type="key", key="Return")
    assert check_action(a).allowed is True
