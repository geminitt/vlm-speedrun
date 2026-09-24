"""Tests for the statistics summary — the core of every number in the project."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.latency_probe import summarize


def test_constant_input_has_no_spread():
    s = summarize([100.0] * 20)
    assert s["median"] == 100.0
    assert s["cv_pct"] == 0.0
    assert s["iqr_pct"] == 0.0


def test_percentiles_are_ordered():
    s = summarize([float(x) for x in range(1, 101)])
    assert s["min"] <= s["p25"] <= s["median"] <= s["p75"] <= s["p95"] <= s["max"]


def test_median_is_not_pulled_by_a_single_outlier():
    """Why the project reports the median instead of the mean."""
    xs = [100.0] * 19 + [10_000.0]
    s = summarize(xs)
    assert s["median"] == 100.0
    assert s["mean"] > 500.0


def test_cv_grows_with_spread():
    narrow = summarize([99.0, 100.0, 101.0] * 7)
    wide = summarize([50.0, 100.0, 150.0] * 7)
    assert wide["cv_pct"] > narrow["cv_pct"]


def test_single_element_does_not_crash():
    s = summarize([42.0])
    assert s["median"] == 42.0 and s["std"] == 0.0
