"""Prune image tokens before they reach the language model.

SmolVLM (Idefics3) data flow:
    image -> vision encoder -> (tiles, 729, 1152)
          -> connector (pixel shuffle) -> (tiles, 81, 2048)
          -> spliced into the sequence at the image-token positions -> language model

In the prompt every tile is a run of 81 image tokens, preceded by text tokens that
mark its place: <fake_token_around_image><row_1_col_2>, ..., <global-img>. We
intervene between the connector and the splice: each tile's run is replaced by a
subset of its own tokens, and every text token — the layout markers included —
stays where it was. The language model sees a shorter sequence with the same
structure.
"""
import math

import torch


def _image_embeds(model, inputs):
    """Return the image tokens after the connector, shaped (N, hidden), tile by tile."""
    inner = model.model
    out = inner.get_image_features(inputs["pixel_values"],
                                   inputs.get("pixel_attention_mask"))
    vis = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if vis.shape[-1] != model.config.text_config.hidden_size:
        vis = inner.connector(vis)
    return vis.reshape(-1, vis.shape[-1])


def _pool_shape(k, side):
    """Grid (h, w) with h * w == k that fits a side x side tile, as square as possible."""
    pairs = [(h, k // h) for h in range(1, side + 1) if k % h == 0 and k // h <= side]
    return min(pairs, key=lambda hw: abs(hw[0] - hw[1])) if pairs else None


def select(embeds, keep_ratio, method="uniform", generator=None, grid=None):
    """Select a subset of the image tokens of ONE tile.

    uniform : evenly spaced — preserves spatial coverage of the tile
    pool    : average spatially adjacent regions — discards nothing, only blurs
    random  : random subset — a control arm, to tell whether 'uniform' beats chance
    norm    : keep the tokens with the largest norm — a signal-strength heuristic

    grid : side length of the tile's square token grid (9 for SmolVLM). With it,
           'pool' averages 2-D regions of the grid; without it, runs of consecutive
           tokens. generator : torch.Generator for 'random', so a run is reproducible.
    """
    n = embeds.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    if k >= n:
        return embeds
    if method == "uniform":
        idx = torch.linspace(0, n - 1, k, device=embeds.device).round().long()
        return embeds[idx]
    if method == "pool":
        shape = _pool_shape(k, grid) if grid and grid * grid == n else None
        if shape:
            g = embeds.reshape(grid, grid, -1).permute(2, 0, 1).float()
            pooled = torch.nn.functional.adaptive_avg_pool2d(g, shape)
            return pooled.permute(1, 2, 0).reshape(k, -1).to(embeds.dtype)
        g = max(1, n // k)
        usable = (n // g) * g
        pooled = embeds[:usable].reshape(-1, g, embeds.shape[-1]).mean(1)
        return pooled if usable == n else torch.cat([pooled, embeds[usable:].mean(0, keepdim=True)])
    if method == "random":
        perm = torch.randperm(n, generator=generator, device="cpu").to(embeds.device)
        return embeds[perm[:k].sort().values]
    if method == "norm":
        idx = embeds.float().norm(dim=-1).topk(k).indices.sort().values
        return embeds[idx]
    raise ValueError(f"unknown method: {method}")


def image_runs(ids, img_id):
    """(start, end) of every contiguous run of image tokens — one run per tile."""
    runs, i, is_img = [], 0, (ids == img_id).tolist()
    while i < len(is_img):
        if is_img[i]:
            j = i
            while j < len(is_img) and is_img[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def splice(ids, img_id, image_embeds, text_embeds, keep_ratio, method="uniform",
           generator=None):
    """Rebuild the input embeddings with each tile's image tokens pruned.

    ids          : (L,) token ids of the prompt
    image_embeds : (N, d) image tokens after the connector, tile by tile
    text_embeds  : (L, d) embeddings of ids (image positions are overwritten)
    Returns (embeddings of the new sequence, number of image tokens kept).
    """
    pieces, prev, offset, kept = [], 0, 0, 0
    for start, end in image_runs(ids, img_id):
        n = end - start
        tile = image_embeds[offset:offset + n]
        side = math.isqrt(n)
        sub = select(tile, keep_ratio, method, generator=generator,
                     grid=side if side * side == n else None)
        pieces += [text_embeds[prev:start], sub.to(text_embeds.dtype)]
        prev, offset, kept = end, offset + n, kept + sub.shape[0]
    assert offset == image_embeds.shape[0], "image features do not match the image tokens"
    pieces.append(text_embeds[prev:])
    return torch.cat(pieces, dim=0), kept


def build_inputs(model, inputs, keep_ratio=1.0, method="uniform", seed=0):
    """Build inputs_embeds with image tokens pruned tile by tile.

    Returns (inputs_embeds, attention_mask, number_of_image_tokens_kept).
    With keep_ratio >= 1 it returns None so the caller uses the original path.
    seed makes 'random' selection reproducible, so every timed call on the same
    sample keeps the same tokens.
    """
    if keep_ratio >= 1.0:
        return None, None, None
    ids = inputs["input_ids"][0]
    img_emb = _image_embeds(model, inputs).to(model.dtype)
    txt_emb = model.get_input_embeddings()(ids)
    gen = torch.Generator().manual_seed(seed) if method == "random" else None
    embeds, kept = splice(ids, model.config.image_token_id, img_emb, txt_emb,
                          keep_ratio, method, generator=gen)
    embeds = embeds.unsqueeze(0)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return embeds, mask, kept
