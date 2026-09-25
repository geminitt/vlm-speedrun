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
