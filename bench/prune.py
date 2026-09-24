"""Prune image tokens before they reach the language model.

SmolVLM (Idefics3) data flow:
    image -> vision encoder -> (tiles, 729, 1152)
          -> connector (pixel shuffle) -> (tiles, 81, 2048)
          -> spliced into the sequence at the image-token positions -> language model

We intervene between the connector and the splice: keep a subset of the image
tokens, then build inputs_embeds ourselves. The language model sees a shorter
sequence.
"""
import torch


def _image_embeds(model, inputs):
    """Return the image tokens after the connector, shaped (N, hidden)."""
    inner = model.model
    out = inner.get_image_features(inputs["pixel_values"],
                                   inputs.get("pixel_attention_mask"))
    vis = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if vis.shape[-1] != model.config.text_config.hidden_size:
        vis = inner.connector(vis)
    return vis.reshape(-1, vis.shape[-1])


def select(embeds, keep_ratio, method="uniform", scores=None):
    """Select a subset of image tokens.

    uniform : evenly spaced — preserves spatial coverage of the image
    pool    : average adjacent groups — discards nothing, only blurs
    random  : random subset — a control arm, to tell whether 'uniform' beats chance
    norm    : keep the tokens with the largest norm — a signal-strength heuristic
    """
    n = embeds.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    if k >= n:
        return embeds
    if method == "uniform":
        idx = torch.linspace(0, n - 1, k, device=embeds.device).round().long()
        return embeds[idx]
    if method == "pool":
        g = max(1, n // k)
        usable = (n // g) * g
        pooled = embeds[:usable].reshape(-1, g, embeds.shape[-1]).mean(1)
        return pooled if usable == n else torch.cat([pooled, embeds[usable:].mean(0, keepdim=True)])
    if method == "random":
        idx = torch.randperm(n, device=embeds.device)[:k].sort().values
        return embeds[idx]
    if method == "norm":
        idx = embeds.float().norm(dim=-1).topk(k).indices.sort().values
        return embeds[idx]
    raise ValueError(f"unknown method: {method}")


def build_inputs(model, inputs, keep_ratio=1.0, method="uniform"):
    """Build inputs_embeds with image tokens pruned.

    Returns (inputs_embeds, attention_mask, number_of_image_tokens_kept).
    With keep_ratio >= 1 it returns None so the caller uses the original path.
    """
    if keep_ratio >= 1.0:
        return None, None, None

    ids = inputs["input_ids"][0]
    img_id = model.config.image_token_id
    is_img = ids == img_id

    img_emb = _image_embeds(model, inputs)
    kept = select(img_emb, keep_ratio, method).to(model.dtype)

    txt_emb = model.get_input_embeddings()(ids)

    # Reassemble: keep the text order, replace the whole image-token block with the pruned one
    first = int(is_img.nonzero()[0]) if is_img.any() else 0
    last = int(is_img.nonzero()[-1]) + 1 if is_img.any() else 0
    pieces = [txt_emb[:first], kept, txt_emb[last:]]
    embeds = torch.cat(pieces, dim=0).unsqueeze(0)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return embeds, mask, kept.shape[0]
