"""Compare two separate runs on the SAME sample set.

Used when two configurations cannot be interleaved in one run — for example bf16
versus nf4, because only one model fits on a 6 GB card at a time. We lose the
benefit of interleaving over time but keep the per-sample paired comparison.
"""
import argparse, json, statistics

from bench.metrics import mcnemar


def compare(run_a, run_b, key="baseline"):
    """Paired comparison of configuration `key` in two runs (loaded JSON dicts).

    Accuracy: McNemar on the samples both runs share (majority verdict over rounds).
    Speed: median over samples of (time in A / time in B), so > 1 means B is faster.
    """
    def fold(run):
        acc, lat = {}, {}
        for r in run["records"]:
            if r["config"] == key:
                acc.setdefault(r["sample_id"], []).append(r["correct"])
                lat.setdefault(r["sample_id"], []).append(r["generate_ms"])
        return ({k: sum(v) * 2 >= len(v) for k, v in acc.items()},
                {k: statistics.median(v) for k, v in lat.items()})

    acc_a, lat_a = fold(run_a)
    acc_b, lat_b = fold(run_b)
    common = sorted(set(acc_a) & set(acc_b))
    only_a = sum(acc_a[i] and not acc_b[i] for i in common)
    only_b = sum(acc_b[i] and not acc_a[i] for i in common)
    ratios = sorted(lat_a[i] / lat_b[i] for i in common)
    n = len(ratios)
    return {"n": n,
            "acc_a": 100 * sum(acc_a[i] for i in common) / n,
            "acc_b": 100 * sum(acc_b[i] for i in common) / n,
            "only_a": only_a, "only_b": only_b, "p_value": mcnemar(only_a, only_b),
            "speed_b_over_a": statistics.median(ratios),
            "speed_iqr": [ratios[n // 4], ratios[(3 * n) // 4]],
            "vram_a": run_a.get("peak_vram_mb", 0.0), "vram_b": run_b.get("peak_vram_mb", 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results file A")
    ap.add_argument("--b", required=True, help="results file B")
    ap.add_argument("--key", default="baseline", help="configuration key to compare")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    x = ap.parse_args()

    c = compare(json.load(open(x.a)), json.load(open(x.b)), x.key)
    sp = c["speed_b_over_a"]
    print(f"configuration: {x.key} | {c['n']} shared samples\n")
    print(f"{x.label_a:>12}: {c['acc_a']:5.1f}% correct | VRAM {c['vram_a']:.0f} MB")
    print(f"{x.label_b:>12}: {c['acc_b']:5.1f}% correct | VRAM {c['vram_b']:.0f} MB")
    print(f"\ndifference    : {c['acc_b'] - c['acc_a']:+.1f} points "
          f"(only {x.label_a} correct: {c['only_a']} · only {x.label_b} correct: {c['only_b']} "
          f"· p = {c['p_value']:.4f})")
    print(f"speed         : {x.label_b} runs at {sp:.2f}× the speed of {x.label_a} "
          f"[{c['speed_iqr'][0]:.2f}–{c['speed_iqr'][1]:.2f}] — "
          + (f"{100 * (sp - 1):.0f}% faster" if sp >= 1 else f"{100 * (1 / sp - 1):.0f}% slower"))
    print(f"verdict       : " + ("quality differs significantly" if c["p_value"] < 0.05
                                 else "quality not distinguishable"))


if __name__ == "__main__":
    main()
