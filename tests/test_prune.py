import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.prune import select


def _emb(n=100, d=8):
    return torch.arange(n * d, dtype=torch.float32).reshape(n, d)


def test_keeping_everything_changes_nothing():
    e = _emb()
    assert torch.equal(select(e, 1.0), e)


def test_keeps_the_right_number_of_tokens():
    e = _emb(100)
    for r in (0.1, 0.25, 0.5, 0.75):
        assert select(e, r, "uniform").shape[0] == int(round(100 * r))


def test_uniform_spans_first_to_last():
    e = _emb(100)
    out = select(e, 0.1, "uniform")
    assert torch.equal(out[0], e[0]) and torch.equal(out[-1], e[-1])


def test_pool_averages_each_group():
    e = _emb(100)
    out = select(e, 0.5, "pool")          # merge pairs
    assert out.shape[0] == 50
    assert torch.allclose(out[0], (e[0] + e[1]) / 2)


def test_norm_keeps_the_strongest_tokens():
    e = torch.zeros(10, 4)
    e[3] = 100.0
    e[7] = 50.0
    out = select(e, 0.2, "norm")
    assert out.shape[0] == 2
    assert out.float().norm(dim=-1).min() > 0


def test_random_keeps_the_right_count():
    e = _emb(100)
    out = select(e, 0.3, "random")
    assert out.shape[0] == 30


def test_unknown_method_raises():
    try:
        select(_emb(), 0.5, "does-not-exist")
        assert False, "should have raised"
    except ValueError:
        pass
