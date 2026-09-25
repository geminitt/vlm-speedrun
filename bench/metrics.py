"""Quality metrics.

ChartQA is scored with the benchmark's *relaxed accuracy* (relaxed_correctness): if
both answers parse as numbers, the prediction is correct within 5% relative error;
otherwise the strings must match, ignoring case. The tolerance exists because chart
values are read off the plot by eye. Two variants serve as a sensitivity check:
exact_years (years must match exactly — the benchmark applies the 5% tolerance to
them too, so 2019 counts for 2017) and lenient_match, the project's earlier metric,
which fished the first number out of any sentence.
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
        return float(m.group(0))        # "45%" and "45" give the same value
    except ValueError:
        return None


def relaxed_correctness(prediction, target, max_relative_change=0.05, exact_years=False):
    """ChartQA relaxed accuracy for one answer, as the benchmark defines it.

    Numbers: the whole string must parse as a float, and a trailing "%" divides it by
    100. With exact_years, a target that is a year (1500-2099) must match exactly.
    """
    def to_float(t):
        try:
            return float(t.rstrip("%")) / 100.0 if t.endswith("%") else float(t)
        except ValueError:
            return None
    prediction, target = prediction.strip(), target.strip()
    if exact_years and re.fullmatch(r"(1[5-9]|20)\d\d", target):
        return prediction == target
    p, t = to_float(prediction), to_float(target)
    if p is not None and t:
        return abs(p - t) / abs(t) <= max_relative_change
    return prediction.lower() == target.lower()


def clean_answer(s):
    """Answer extraction before scoring: trim, and drop one trailing full stop.

    Under the project's instruction the model ends short answers with a full stop
    ("Cameroon."). That is punctuation of the reply, not part of the answer.
    """
    s = (s or "").strip()
    return s[:-1].rstrip() if s.endswith(".") else s


def normalize_text(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s%.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .")  # drop surrounding punctuation: "Yes." -> "yes"


def lenient_match(pred, gold, tol=0.05):
    """The project's earlier ChartQA metric, kept only for the sensitivity check.

    More lenient than the benchmark: it takes the FIRST number anywhere in the answer
    ("82.39 billion" counts for 82.39, but so does "1992, 2016" for 2009), drops
    thousands separators, and treats "45%" as 45.
    """
    p_num, g_num = to_number(pred), to_number(gold)
    if g_num is not None and p_num is not None:
        if g_num == 0:
            return abs(p_num) <= tol
        return abs(p_num - g_num) / abs(g_num) <= tol
    return normalize_text(pred) == normalize_text(gold)


def score_chartqa(pred, gold, metric="relaxed"):
    """Score one ChartQA answer after answer cleaning. metric: relaxed | exact_years | lenient."""
    pred = clean_answer(pred)
    if metric == "lenient":
        return lenient_match(pred, gold)
    return relaxed_correctness(pred, gold, exact_years=(metric == "exact_years"))


def relaxed_accuracy(preds, golds):
    assert len(preds) == len(golds)
    if not preds:
        return 0.0
    return sum(score_chartqa(p, g) for p, g in zip(preds, golds)) / len(preds)


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


def robust_cv(xs):
    """Noise as a robust coefficient of variation, in %: IQR / 1.349 / median.

    For a normal distribution the IQR spans 1.349 standard deviations, so this
    estimates the CV while ignoring rare outliers — one slow run can double the
    plain CV (standard deviation / mean) of sixty otherwise steady ones.
    """
    import statistics
    q = statistics.quantiles(sorted(xs), n=4)
    return 100 * (q[2] - q[0]) / 1.349 / statistics.median(xs)


def bootstrap_ci(values, stat=None, n_boot=10_000, seed=0, level=0.95):
    """Percentile bootstrap interval for a statistic of independent samples.

    Resamples the samples with replacement n_boot times and takes the central
    `level` share of the recomputed statistic. stat defaults to the mean.
    """
    import random
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    boots = sorted(stat(rng.choices(values, k=len(values))) for _ in range(n_boot))
    lo = boots[int((1 - level) / 2 * n_boot)]
    hi = boots[int((1 + level) / 2 * n_boot) - 1]
    return lo, hi


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order.

    With m comparisons, the i-th smallest p-value is multiplied by (m - i), and the
    sequence is made non-decreasing. Controls the chance of any false positive at
    alpha, and rejects at least as much as plain Bonferroni.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = round(running, 12)
    return adjusted


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
    Normalisation follows the official metric: lowercase and collapse whitespace,
    nothing else (punctuation counts). Pass the prediction through clean_answer first.
    """
    if isinstance(golds, str):
        golds = [golds]
    norm = lambda s: " ".join((s or "").lower().split())
    p = norm(pred)
    best = 0.0
    for g in golds:
        g = norm(g)
        if not p and not g:
            best = max(best, 1.0); continue
        d = levenshtein(p, g)
        best = max(best, 1 - d / max(len(p), len(g), 1))
    return best if best >= threshold else 0.0
