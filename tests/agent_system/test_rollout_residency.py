from agent_system.multi_turn_rollout.rollout_loop import rollout_residency_scope


class _FakeWorkerGroup:
    def __init__(self):
        self.begin_count = 0
        self.end_count = 0
        self.generate_count = 0
        self.resident = False

    def begin_rollout(self):
        self.begin_count += 1
        self.resident = True

    def end_rollout(self):
        self.end_count += 1
        self.resident = False

    def generate_sequences(self):
        if self.resident:
            self.generate_count += 1


def test_rollout_residency_scope_starts_and_ends_each_group_once():
    used = _FakeWorkerGroup()
    unused = _FakeWorkerGroup()

    with rollout_residency_scope({"used": used, "unused": unused}):
        used.generate_sequences()
        assert used.begin_count == 1
        assert unused.begin_count == 1
        assert used.generate_count == 1
        assert unused.generate_count == 0
        assert used.end_count == 0
        assert unused.end_count == 0

    assert used.end_count == 1
    assert unused.end_count == 1


def test_rollout_residency_scope_releases_groups_on_error():
    group = _FakeWorkerGroup()

    try:
        with rollout_residency_scope(group):
            raise RuntimeError("generation failed")
    except RuntimeError:
        pass

    assert group.begin_count == 1
    assert group.end_count == 1


def test_rollout_residency_scope_deduplicates_shared_group():
    group = _FakeWorkerGroup()

    with rollout_residency_scope({"agent_a": group, "agent_b": group}):
        pass

    assert group.begin_count == 1
    assert group.end_count == 1
