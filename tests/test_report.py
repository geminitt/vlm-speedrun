import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.report import delta_p


def test_delta_uses_only_the_samples_both_configurations_share():
    # A ran on four questions, B only on the first two: the difference must compare
    # the same two questions, not A's four with B's two.
    recs = [{"config": "A", "sample_id": i, "correct": ok} for i, ok in enumerate([1, 1, 0, 0])]
    recs += [{"config": "B", "sample_id": i, "correct": ok} for i, ok in enumerate([1, 0])]
    d, p = delta_p(recs, "A", "B")
    assert d == -50.0                     # A: 2/2 on the shared questions, B: 1/2
    assert p == 1.0                       # one discordant pair


def test_docvqa_verdict_states_the_split_limit_and_assigns_no_cause():
    # The check compares a validation-split sample with a published test-split score
    # under a different evaluation setup: an interval that excludes that score shows a
    # difference beyond sampling error, not that the pipeline is at fault, nor why.
    from bench.report import ROOT, values
    verdict = values(ROOT / "results")["docvqa_verdict"]
    assert "split" in verdict
    assert "systematic" not in verdict and "fault" not in verdict
    for published, mean, lo, hi in ((81.6, 71.7, 67.2, 76.2), (81.6, 80.0, 76.0, 84.0)):
        from bench.report import docvqa_verdict
        text = docvqa_verdict(published, mean, lo, hi)
        assert "split" in text and "systematic" not in text and "fault" not in text
