import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.prune import select, splice


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


# --- splicing pruned tiles back into the sequence -------------------------------

IMG, TXT = 99, 1          # image-token id and a filler text-token id for the tests


def _sequence(n_tiles=3, per_tile=81, markers=2):
    """Text markers around every tile, like SmolVLM's <row_i_col_j> tokens."""
    ids = [TXT] * 4
    for t in range(n_tiles):
        ids += [100 + t] * markers + [IMG] * per_tile
    return torch.tensor(ids + [TXT] * 5)


def test_splice_keeps_every_text_token_in_place():
    ids = _sequence()
    d = 4
    text = torch.arange(len(ids) * d, dtype=torch.float32).reshape(len(ids), d)
    image = torch.randn(3 * 81, d)
    out, kept = splice(ids, IMG, image, text, 0.25, "uniform")
    n_text = int((ids != IMG).sum())
    assert kept == 3 * 20                               # round(81 * 0.25) per tile
    assert out.shape[0] == n_text + kept
    # every text embedding survives, in the original order
    text_rows = text[ids != IMG]
    found = [any(torch.equal(r, o) for o in out) for r in text_rows]
    assert all(found)


def test_splice_puts_each_tiles_tokens_after_its_own_markers():
    ids = _sequence(n_tiles=2, per_tile=81, markers=1)
    d = 2
    text = torch.full((len(ids), d), -1.0)
    image = torch.cat([torch.zeros(81, d), torch.ones(81, d)])   # tile 0 = 0s, tile 1 = 1s
    out, _ = splice(ids, IMG, image, text, 0.5, "uniform")
    # layout: 4 text, marker, 40 of tile 0, marker, 40 of tile 1, 5 text
    tile1_block = out[4 + 1 + 40 + 1: 4 + 1 + 40 + 1 + 40]
    assert torch.all(tile1_block == 1.0)
    assert torch.all(out[4 + 1: 4 + 1 + 40] == 0.0)


def test_keep_ratio_one_returns_the_sequence_unchanged():
    ids = _sequence(n_tiles=1)
    text = torch.randn(len(ids), 3)
    image = torch.randn(81, 3)
    out, kept = splice(ids, IMG, image, text, 1.0, "uniform")
    expected = text.clone()
    expected[ids == IMG] = image
    assert kept == 81 and torch.equal(out, expected)


def test_pool_averages_spatial_neighbours_within_a_tile():
    # each token carries its (row, col) on the 9x9 grid of one tile
    grid = torch.tensor([[r, c] for r in range(9) for c in range(9)], dtype=torch.float32)
    out = select(grid, 20 / 81, "pool", grid=9)            # 20 = 4 x 5 pooled regions
    assert out.shape[0] == 20
    rows, cols = out[:, 0].reshape(4, 5), out[:, 1].reshape(4, 5)
    assert torch.all(rows[1:] > rows[:-1])                  # regions move down row by row
    assert torch.all(cols[:, 1:] > cols[:, :-1])            # and right column by column


def test_random_selection_is_reproducible_with_a_seed():
    e = _emb(81)
    a = select(e, 0.25, "random", generator=torch.Generator().manual_seed(7))
    b = select(e, 0.25, "random", generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
