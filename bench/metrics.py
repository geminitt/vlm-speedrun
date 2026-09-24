"""Quality metrics.

ChartQA is scored with *relaxed accuracy*: a numeric answer counts as correct if
its relative error is at most 5%; a text answer must match after normalisation.
The tolerance exists because chart questions usually require reading values off
the plot by eye, so demanding an exact match would be unreasonable.
"""
import re

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def to_number(s):
    """Extract the first number in a string. Returns None if there is none."""
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
    if "%" in s:  # "45%" and "45" are treated as the same value
        pass
    return v


def normalize_text(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s%.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .")  # drop surrounding punctuation: "Yes." -> "yes"


def relaxed_match(pred, gold, tol=0.05):
    """True if the prediction matches the answer under ChartQA relaxed accuracy."""
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
    """95% confidence interval for a proportion (Wilson method).

    Used instead of the normal approximation because it stays valid when n is
    small or the proportion is close to 0 or 1. Returns (lower, upper).
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy_ci(records):
    """95% Wilson interval for accuracy, counting each sample once.

    Replicate rounds of the same sample are not independent observations: greedy
    decoding gives the same verdict every round, so counting records instead of
    samples would shrink the interval by about sqrt(rounds). Each sample contributes
    its mean correctness over the rounds. Returns ((lower, upper), n_samples).
    """
    from collections import defaultdict
    per = defaultdict(list)
    for r in records:
        sid = r["sample_id"] if isinstance(r, dict) else r.sample_id
        ok = r["correct"] if isinstance(r, dict) else r.correct
        per[sid].append(float(ok))
    n = len(per)
    k = sum(sum(v) / len(v) for v in per.values())
    return wilson_interval(k, n), n


def mcnemar(b, c):
    """McNemar test for two classifiers evaluated on the SAME samples.

    b = number of samples A gets right and B gets wrong
    c = number of samples A gets wrong and B gets right
    Samples both get right or both get wrong carry no comparative information and
    are left out — which is exactly why this test is more powerful than comparing
    two independent confidence intervals.

    Returns the two-sided p-value, computed exactly from the binomial with p = 0.5.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_accuracy(records, key_a, key_b):
    """Compare the accuracy of two configurations sample by sample."""
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
        a_ok = sum(d[key_a]) * 2 >= len(d[key_a])   # majority over rounds
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
    """Edit distance between two strings."""
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
    """ANLS — the standard DocVQA metric.

    For each reference answer, compute the similarity 1 - distance / max_length.
    Take the best; below the threshold the score is 0.

    ANLS is used instead of exact match because document answers are text read off
    an image, so a single OCR-level character error should not count as fully wrong.
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
