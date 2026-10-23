import pytest

from app.core.timing import StageTimer


def test_stages_are_recorded_in_order_without_overlap():
    timer = StageTimer()
    for name in ("a", "b", "c"):
        with timer.stage(name):
            sum(range(1000))
    total = timer.elapsed_ns()
    stages = timer.stages
    assert [s.name for s in stages] == ["a", "b", "c"]
    end = 0
    for s in stages:
        assert isinstance(s.offset_ns, int) and isinstance(s.duration_ns, int)
        assert s.offset_ns >= end and s.duration_ns >= 0
        end = s.offset_ns + s.duration_ns
    assert sum(s.duration_ns for s in stages) <= end <= total


def test_a_stage_that_raises_is_not_recorded():
    timer = StageTimer()
    with pytest.raises(ValueError):
        with timer.stage("fails"):
            raise ValueError
    with timer.stage("next"):
        pass
    assert [s.name for s in timer.stages] == ["next"]


def test_stages_do_not_nest():
    timer = StageTimer()
    with pytest.raises(RuntimeError):
        with timer.stage("outer"):
            with timer.stage("inner"):
                pass
