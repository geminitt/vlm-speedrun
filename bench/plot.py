"""Plot the accuracy-versus-speed trade-off.

Presentation rules:
  - colour encodes the LEVER FAMILY (baseline / fewer tiles / token pruning), never rank
  - every point carries a direct label, because the palette has one low-contrast slot
  - error bars are the 95% Wilson confidence interval of accuracy
  - grid and axes recede, the data comes forward
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2dd",
                  series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#34332f",
                 series=("#3987e5", "#d95926", "#199e70")),
}
FAMILIES = ["Baseline", "Fewer image tiles", "Prune tokens after encoding"]
# Labels are staggered like steps. Points in the left half are always labelled to
# the right so they are not clipped at the edge, and vice versa.
RIGHT = [(13, 12), (13, -12), (13, 34), (13, -32), (13, 56), (13, -52)]
LEFT = [(-13, 10), (-13, -6), (-13, 24), (-13, -20)]


def family_of(key):
    if key.startswith("baseline"):
        return 0
    if "nosplit" in key or "edge" in key:
        return 1
    return 2


def load(path):
    r = json.load(open(path))
    sp = r.get("paired_speedup_vs_baseline", {})
    rows = []
    for key, c in r["configs"].items():
        rows.append(dict(
            key=key.split("(")[0],
            fam=family_of(key),
            acc=100 * c["accuracy"],
            lo=100 * c["accuracy_ci95"][0],
            hi=100 * c["accuracy_ci95"][1],
            speed=sp.get(key, {}).get("median_speedup", 1.0),
            tokens=c["image_tokens_median"],
        ))
    rows = sorted(rows, key=lambda d: d["speed"])
    xs = [d["speed"] for d in rows]
    mid = (min(xs) + max(xs)) / 2
    for i, d in enumerate(rows):
        d["slot"], d["x_mid"] = i, mid
    return r, rows


def draw(path, out, theme="light"):
    r, rows = load(path)
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=170)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])

    for f, name in enumerate(FAMILIES):
        pts = [d for d in rows if d["fam"] == f]
        if not pts:
            continue
        ax.errorbar([d["speed"] for d in pts], [d["acc"] for d in pts],
                    yerr=[[d["acc"] - d["lo"] for d in pts],
                          [d["hi"] - d["acc"] for d in pts]],
                    fmt="o", ms=9, lw=0, elinewidth=1.6, capsize=3,
                    color=t["series"][f], ecolor=t["series"][f], alpha=0.95,
                    markeredgecolor=t["surface"], markeredgewidth=1.5, label=name)
        for d in pts:                       # direct labels, staggered so they do not overlap
            if d["speed"] < d["x_mid"]:
                dx, dy = RIGHT[d["slot"] % len(RIGHT)]; ha = "left"
            else:
                dx, dy = LEFT[d["slot"] % len(LEFT)]; ha = "right"
            ax.annotate(f"{d['key']} · {d['tokens']:.0f} tok",
                        (d["speed"], d["acc"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color=t["ink2"], ha=ha,
                        arrowprops=dict(arrowstyle="-", color=t["grid"], lw=0.8,
                                        shrinkA=0, shrinkB=6))

    base = next(d for d in rows if d["fam"] == 0)
    ax.axhline(base["acc"], color=t["grid"], lw=1.2, ls="--", zorder=0)
    ax.axhspan(base["lo"], base["hi"], color=t["grid"], alpha=0.5, zorder=0)
    ax.annotate("baseline confidence interval", (ax.get_xlim()[1], base["hi"]),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=8, color=t["ink2"])

    ax.set_xlabel("Speedup over baseline (×, paired)", color=t["ink2"], fontsize=10)
    ax.set_ylabel("ChartQA accuracy (%)", color=t["ink2"], fontsize=10)
    ax.set_title(f"Speed versus quality — {r['model'].split('/')[-1]}"
                 f" on a 6 GB GPU\n{r['samples']} samples × {r['rounds']} rounds",
                 color=t["ink"], fontsize=12, loc="left", pad=12)
    ax.grid(True, color=t["grid"], lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
                    bbox_to_anchor=(0.5, -0.16))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.margins(x=0.13, y=0.18)
    fig.tight_layout()
    fig.savefig(out, facecolor=t["surface"])
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gate2_sweep.json")
    ap.add_argument("--stem", default="results/tradeoff")
    a = ap.parse_args()
    for theme in ("light", "dark"):
        draw(a.results, f"{a.stem}_{theme}.png", theme)
