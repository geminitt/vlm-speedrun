"""Thước đo chất lượng cho ChartQA.

ChartQA chấm bằng *relaxed accuracy*: đáp án số được coi là đúng nếu sai lệch
tương đối không quá 5%; đáp án chữ thì so khớp sau khi chuẩn hoá.
Lý do: câu hỏi trên biểu đồ thường yêu cầu đọc giá trị bằng mắt, nên đòi khớp
tuyệt đối là vô lý.
"""
import re

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def to_number(s):
    """Rút số đầu tiên trong chuỗi. Trả về None nếu không có số."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    m = _NUM.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if "%" in s:  # "45%" và "45" coi như cùng một giá trị
        pass
    return v


def normalize_text(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s%.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .")  # bỏ dấu câu ở hai đầu: "Yes." -> "yes"


def relaxed_match(pred, gold, tol=0.05):
    """True nếu dự đoán khớp đáp án theo chuẩn relaxed accuracy của ChartQA."""
    p_num, g_num = to_number(pred), to_number(gold)
    if g_num is not None and p_num is not None:
        if g_num == 0:
            return abs(p_num) <= tol
        return abs(p_num - g_num) / abs(g_num) <= tol
    return normalize_text(pred) == normalize_text(gold)


def relaxed_accuracy(preds, golds, tol=0.05):
    assert len(preds) == len(golds)
    if not preds:
        return 0.0
    return sum(relaxed_match(p, g, tol) for p, g in zip(preds, golds)) / len(preds)


def wilson_interval(k, n, z=1.96):
    """Khoảng tin cậy 95% cho một tỉ lệ (phương pháp Wilson).

    Dùng thay cho công thức chuẩn vì nó vẫn đúng khi n nhỏ hoặc tỉ lệ gần 0/1.
    Trả về (cận dưới, cận trên).
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(b, c):
    """Kiểm định McNemar cho hai bộ phân loại chạy trên CÙNG tập mẫu.

    b = số mẫu A đúng nhưng B sai
    c = số mẫu A sai nhưng B đúng
    Các mẫu cả hai cùng đúng hoặc cùng sai không mang thông tin so sánh, nên
    bị loại khỏi phép kiểm — đó chính là chỗ phép kiểm này mạnh hơn việc so
    hai khoảng tin cậy độc lập.

    Trả về p-value hai phía, tính chính xác bằng phân phối nhị thức với p=0,5.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_accuracy(records, key_a, key_b):
    """So sánh độ chính xác của hai cấu hình theo từng mẫu."""
    from collections import defaultdict
    per = defaultdict(dict)
    for r in records:
        cfg = r["config"] if isinstance(r, dict) else r.config
        sid = r["sample_id"] if isinstance(r, dict) else r.sample_id
        ok = r["correct"] if isinstance(r, dict) else r.correct
        per[sid].setdefault(cfg, []).append(ok)
    b = c = both = neither = 0
    for sid, d in per.items():
        if key_a not in d or key_b not in d:
            continue
        a_ok = sum(d[key_a]) * 2 >= len(d[key_a])   # đa số các vòng
        b_ok = sum(d[key_b]) * 2 >= len(d[key_b])
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
        elif a_ok:
            both += 1
        else:
            neither += 1
    return {"n_pairs": b + c + both + neither,
            "only_a_correct": b, "only_b_correct": c,
            "both": both, "neither": neither,
            "p_value": mcnemar(b, c)}


def levenshtein(a, b):
    """Khoảng cách chỉnh sửa giữa hai chuỗi."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, golds, threshold=0.5):
    """ANLS — thước đo chuẩn của DocVQA.

    Với mỗi đáp án chuẩn, tính độ tương đồng 1 − khoảng_cách/độ_dài_lớn_nhất.
    Lấy giá trị lớn nhất; nếu dưới ngưỡng thì tính 0 điểm.

    Lý do dùng ANLS thay vì so khớp tuyệt đối: đáp án trong tài liệu là chữ do người
    đọc từ ảnh, nên sai một ký tự do OCR không nên bị tính là sai hoàn toàn.
    """
    if isinstance(golds, str):
        golds = [golds]
    p = normalize_text(pred)
    best = 0.0
    for g in golds:
        g = normalize_text(g)
        if not p and not g:
            best = max(best, 1.0); continue
        d = levenshtein(p, g)
        best = max(best, 1 - d / max(len(p), len(g), 1))
    return best if best >= threshold else 0.0
