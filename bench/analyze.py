"""Analyse recorded results: paired comparison of both speed and accuracy.

Runs on existing JSON files; no model needs to be re-run.
"""
import argparse, json

from bench.metrics import paired_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gate2_edge_sweep.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    r = json.load(open(a.results))
    recs = r["records"]
    keys = list(r["configs"])
    base = keys[0]
    sp = r.get("paired_speedup_vs_baseline", {})

    print(f"{r['model']} | {r['samples']} samples × {r['rounds']} rounds "
          f"| compared against: {base}\n")
    head = (f"{'config':<22}{'tokens':>7}{'acc':>7}{'Δ pts':>8}"
            f"{'only base':>10}{'only new':>9}{'p':>9}  verdict")
    print(head); print("-" * len(head))
    acc_base = 100 * r["configs"][base]["accuracy"]
    for k in keys:
        c = r["configs"][k]
        acc = 100 * c["accuracy"]
        tok = c["image_tokens_median"]
        if k == base:
            print(f"{k.split('(')[0]:<22}{tok:>7.0f}{acc:>6.1f}%{'—':>8}"
                  f"{'—':>10}{'—':>9}{'—':>9}  (reference)")
            continue
        pa = paired_accuracy(recs, base, k)
        p = pa["p_value"]
        speed = sp.get(k, {}).get("median_speedup", float("nan"))
        verdict = ("quality significantly WORSE" if p < a.alpha and pa["only_a_correct"] > pa["only_b_correct"]
                   else "quality significantly BETTER" if p < a.alpha
                   else "not distinguishable")
        print(f"{k.split('(')[0]:<22}{tok:>7.0f}{acc:>6.1f}%{acc-acc_base:>+8.1f}"
              f"{pa['only_a_correct']:>10}{pa['only_b_correct']:>9}{p:>9.4f}"
              f"  {speed:.2f}× · {verdict}")

    print("\nNote: p is a McNemar test on the discordant pairs. "
          f"Threshold used here: {a.alpha}.")


if __name__ == "__main__":
    main()
