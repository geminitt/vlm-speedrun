"""Cắt bớt token ảnh trước khi đưa vào mô hình ngôn ngữ.

Luồng gốc của SmolVLM (Idefics3):
    ảnh -> bộ mã hoá thị giác -> (số_ô, 729, 1152)
        -> connector (pixel shuffle) -> (số_ô, 81, 2048)
        -> chèn vào chuỗi tại các vị trí token ảnh -> mô hình ngôn ngữ

Ta chen vào giữa bước connector và bước chèn: giữ lại một phần token ảnh,
rồi tự dựng inputs_embeds. Mô hình ngôn ngữ khi đó thấy một chuỗi ngắn hơn.
"""
import torch


def _image_embeds(model, inputs):
    """Trả về token ảnh sau connector, dạng (N, hidden)."""
    inner = model.model
    out = inner.get_image_features(inputs["pixel_values"],
                                   inputs.get("pixel_attention_mask"))
    vis = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if vis.shape[-1] != model.config.text_config.hidden_size:
        vis = inner.connector(vis)
    return vis.reshape(-1, vis.shape[-1])


def select(embeds, keep_ratio, method="uniform", scores=None):
    """Chọn tập con token ảnh.

    uniform : lấy cách đều — giữ nguyên độ phủ không gian của ảnh
    pool    : gộp trung bình từng nhóm liền kề — không vứt thông tin, chỉ làm nhoè
    random  : lấy ngẫu nhiên — dùng làm đối chứng, để biết 'uniform' có hơn may rủi không
    norm    : giữ token có chuẩn lớn nhất — dựa vào độ mạnh tín hiệu
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
    raise ValueError(f"phương pháp lạ: {method}")


def build_inputs(model, inputs, keep_ratio=1.0, method="uniform"):
    """Dựng inputs_embeds đã cắt token ảnh.

    Trả về (inputs_embeds, attention_mask, số_token_ảnh_còn_lại).
    Khi keep_ratio >= 1 thì trả về None để phía gọi dùng đường chạy gốc.
    """
    if keep_ratio >= 1.0:
        return None, None, None

    ids = inputs["input_ids"][0]
    img_id = model.config.image_token_id
    is_img = ids == img_id

    img_emb = _image_embeds(model, inputs)
    kept = select(img_emb, keep_ratio, method).to(model.dtype)

    txt_emb = model.get_input_embeddings()(ids)

    # Ghép lại: giữ nguyên thứ tự văn bản, thay toàn bộ khối token ảnh bằng khối đã cắt
    first = int(is_img.nonzero()[0]) if is_img.any() else 0
    last = int(is_img.nonzero()[-1]) + 1 if is_img.any() else 0
    pieces = [txt_emb[:first], kept, txt_emb[last:]]
    embeds = torch.cat(pieces, dim=0).unsqueeze(0)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    return embeds, mask, kept.shape[0]
