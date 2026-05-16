from src.agent.reasoner import StubReasoner


def test_stub_emits_plan_for_open_folder():
    r = StubReasoner()
    actions = []
    for i in range(5):
        a, _ = r.next_action("Open My Documents folder", "desktop", i)
        actions.append(a.type)
    # Should reach task_complete within the planned steps
    assert "task_complete" in actions


def test_stub_resets_state():
    r = StubReasoner()
    for i in range(3):
        r.next_action("test", "obs", i)
    assert len(r.history) == 3
    r.reset()
    assert r.history == []
