"""Compare two separate runs on the SAME sample set.

Used when two configurations cannot be interleaved in one run — for example bf16
versus nf4, because only one model fits on a 6 GB card at a time. We lose the
benefit of interleaving over time but keep the per-sample paired comparison.
"""
import argparse, json, statistics

from bench.metrics import mcnemar


def records(path, key):
    r = json.load(open(path))
    recs = [x for x in r["records"] if x["config"] == key]
    return r, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results file A")
    ap.add_argument("--b", required=True, help="results file B")
    ap.add_argument("--key", default="baseline", help="configuration key to compare")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    x = ap.parse_args()

    ra, reca = records(x.a, x.key)
    rb, recb = records(x.b, x.key)

    def fold(recs):
        acc, lat = {}, {}
        for r in recs:
            acc.setdefault(r["sample_id"], []).append(r["correct"])
            lat.setdefault(r["sample_id"], []).append(r["generate_ms"])
        return ({k: sum(v) * 2 >= len(v) for k, v in acc.items()},
                {k: statistics.median(v) for k, v in lat.items()})

    acc_a, lat_a = fold(reca)
    acc_b, lat_b = fold(recb)
    common = sorted(set(acc_a) & set(acc_b))
    only_a = sum(acc_a[i] and not acc_b[i] for i in common)
    only_b = sum(acc_b[i] and not acc_a[i] for i in common)
    p = mcnemar(only_a, only_b)
    ratios = sorted(lat_a[i] / lat_b[i] for i in common)
    n = len(ratios)

    pa = 100 * sum(acc_a[i] for i in common) / len(common)
    pb = 100 * sum(acc_b[i] for i in common) / len(common)
    print(f"configuration: {x.key} | {len(common)} shared samples\n")
    print(f"{x.label_a:>12}: {pa:5.1f}% correct | VRAM {ra.get('peak_vram_mb', 0):.0f} MB")
    print(f"{x.label_b:>12}: {pb:5.1f}% correct | VRAM {rb.get('peak_vram_mb', 0):.0f} MB")
    print(f"\ndifference    : {pb - pa:+.1f} points "
          f"(only {x.label_a} correct: {only_a} · only {x.label_b} correct: {only_b} · p = {p:.4f})")
    print(f"speed         : {x.label_b} faster by {statistics.median(ratios):.2f}× "
          f"[{ratios[n//4]:.2f}–{ratios[(3*n)//4]:.2f}]")
    print(f"verdict       : " + ("quality differs significantly" if p < 0.05
                                 else "quality not distinguishable"))


if __name__ == "__main__":
    main()
