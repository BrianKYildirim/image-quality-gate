import itertools
import math

import numpy as np
import pytest

from app.core.timing import StageTimer
from app.services.quality import (
    Thresholds,
    assess,
    assess_traced,
    evaluate,
    variance_of_laplacian,
)

THR = Thresholds(blur_min=140.0, bright_min=20, bright_max=235)


def uniform(value: int) -> np.ndarray:
    # Brightness is exactly `value` and the Laplacian variance exactly 0.0.
    return np.full((32, 32), value, dtype=np.uint8)


@pytest.fixture
def textured() -> np.ndarray:
    return np.random.default_rng(0).integers(30, 220, (64, 64), dtype=np.uint8)


def test_assess_keeps_its_return_shape(textured):
    res = assess(textured, THR)
    assert list(res) == ["blur_score", "brightness", "is_ok"]
    assert type(res["blur_score"]) is float
    assert type(res["brightness"]) is float
    assert type(res["is_ok"]) is bool


def test_scores_come_from_the_returned_laplacian(textured):
    result = assess_traced(textured, THR, StageTimer(), keep_laplacian=True)
    assert result.laplacian is not None
    assert result.blur_score == float(result.laplacian.var())
    assert result.blur_score == variance_of_laplacian(textured)
    assert result.brightness == float(textured.mean())


def test_laplacian_is_only_kept_when_asked_for(textured):
    kept = assess_traced(textured, THR, StageTimer(), keep_laplacian=True)
    released = assess_traced(textured, THR, StageTimer())
    assert released.laplacian is None
    assert (released.blur_score, released.brightness, released.is_ok) == (
        kept.blur_score,
        kept.brightness,
        kept.is_ok,
    )


def test_blur_exactly_at_blur_min_passes_and_just_above_fails(textured):
    score = assess(textured, THR)["blur_score"]
    assert score > 0

    at = assess_traced(textured, Thresholds(score, 0, 255), StageTimer())
    assert at.checks[0].value == score
    assert at.checks[0].passed is True
    assert at.is_ok is True

    above = Thresholds(math.nextafter(score, math.inf), 0, 255)
    just_above = assess_traced(textured, above, StageTimer())
    assert just_above.checks[0].passed is False
    assert just_above.is_ok is False


def test_zero_blur_at_zero_blur_min_passes():
    assert assess(uniform(100), Thresholds(0.0, 0, 255))["is_ok"] is True
    tiny = Thresholds(math.nextafter(0.0, math.inf), 0, 255)
    assert assess(uniform(100), tiny)["is_ok"] is False


@pytest.mark.parametrize(
    "value, expected",
    [
        (19, [True, False, True]),
        (20, [True, True, True]),
        (21, [True, True, True]),
        (234, [True, True, True]),
        (235, [True, True, True]),
        (236, [True, True, False]),
    ],
)
def test_brightness_bounds_are_inclusive(value, expected):
    result = assess_traced(uniform(value), Thresholds(0.0, 20, 235), StageTimer())
    assert result.brightness == float(value)
    assert [c.passed for c in result.checks] == expected
    assert result.is_ok is all(expected)


def test_brightness_just_outside_the_bounds_fails():
    thr = Thresholds(0.0, 20, 235)
    below = evaluate(0.0, math.nextafter(20.0, -math.inf), thr)
    above = evaluate(0.0, math.nextafter(235.0, math.inf), thr)
    assert [c.passed for c in below] == [True, False, True]
    assert [c.passed for c in above] == [True, True, False]


def test_checks_report_comparator_threshold_and_value():
    blur, bright = evaluate(150.5, 64.25, THR)[0], evaluate(150.5, 64.25, THR)[1:]
    assert (blur.name, blur.measurement, blur.comparator) == (
        "blur_min",
        "blur_score",
        ">=",
    )
    assert (blur.threshold, blur.value) == (140.0, 150.5)
    assert [(c.name, c.comparator, c.threshold, c.value) for c in bright] == [
        ("bright_min", ">=", 20, 64.25),
        ("bright_max", "<=", 235, 64.25),
    ]


def test_is_ok_is_the_conjunction_of_checks_and_the_original_rule():
    blurs = [0.0, 139.999, 140.0, 140.001, 1e6, math.nan]
    brights = [0.0, 19.99, 20.0, 128.0, 235.0, 235.01, 255.0, math.nan]
    for blur, bright in itertools.product(blurs, brights):
        checks = evaluate(blur, bright, THR)
        original = (blur >= THR.blur_min) and (
            THR.bright_min <= bright <= THR.bright_max
        )
        assert all(c.passed for c in checks) == original, (blur, bright)


def test_assess_traced_records_decision_stages_in_order(textured):
    timer = StageTimer()
    assess_traced(textured, THR, timer)
    assert [s.name for s in timer.stages] == ["laplacian", "statistics", "decision"]
