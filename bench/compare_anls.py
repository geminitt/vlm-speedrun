"""Compare two DocVQA runs (bench/sanity_docvqa.py) on the same samples.

ANLS is a score between 0 and 1, not a right/wrong verdict, so the paired test is a
sign test: on how many samples does B score higher than A, and on how many lower?
Samples where both score the same carry no information, as in McNemar's test.
"""
import argparse, json

from bench.metrics import bootstrap_ci, mcnemar


def compare(rows_a, rows_b):
    a = {r["sample_id"]: r["anls"] for r in rows_a}
    b = {r["sample_id"]: r["anls"] for r in rows_b}
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(ids),
            "mean_a": 100 * sum(a[i] for i in ids) / len(ids),
            "mean_b": 100 * sum(b[i] for i in ids) / len(ids),
            "diff": 100 * sum(diffs) / len(ids),
            "diff_ci95": [100 * v for v in bootstrap_ci(diffs)],
            "b_better": better, "b_worse": worse,
            "p_value": mcnemar(better, worse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results/sanity_docvqa.json")
    ap.add_argument("--b", default="results/sanity_docvqa_en.json")
    ap.add_argument("--label-a", default="Vietnamese instruction")
    ap.add_argument("--label-b", default="English instruction")
    x = ap.parse_args()
    c = compare(json.load(open(x.a))["rows"], json.load(open(x.b))["rows"])
    print(f"{c['n']} shared samples")
    print(f"{x.label_a:>24}: ANLS {c['mean_a']:.1f}")
    print(f"{x.label_b:>24}: ANLS {c['mean_b']:.1f}")
    print(f"difference B - A: {c['diff']:+.1f} points, 95% CI "
          f"[{c['diff_ci95'][0]:+.1f}, {c['diff_ci95'][1]:+.1f}]")
    print(f"sign test: B higher on {c['b_better']}, lower on {c['b_worse']} "
          f"-> p = {c['p_value']:.3f}")


if __name__ == "__main__":
    main()
