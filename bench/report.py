"""Generate README.md from README.template.md and the result files.

Every number in the README comes from here. The template holds the prose; each
{{name}} placeholder is filled from results/*.json, so no figure is ever copied by
hand from a terminal, and a result file that changes without the README changing
is caught:

    python -m bench.report --write     # regenerate README.md
    python -m bench.report --check     # fail if README.md is out of date (CI runs this)
"""
import argparse, json, re, statistics, sys
from pathlib import Path

from bench.compare_anls import compare as compare_anls
from bench.compare_runs import compare as compare_runs
from bench.metrics import (accuracy_ci, bootstrap_ci, clean_answer, holm, mcnemar_power,
                           paired_accuracy, robust_cv, samples_for_power, score_chartqa)

ROOT = Path(__file__).resolve().parents[1]
TOKENS_PER_TILE = 81   # SmolVLM: 729 SigLIP patches per 384-pixel tile, pixel-shuffled down to 81 tokens
TEMPLATE, README = ROOT / "README.template.md", ROOT / "README.md"

LABELS = {   # configuration key prefix -> README label
    "baseline": "Baseline (13 tiles, longest edge 1536)",
    "edge1152": "Longest edge 1152", "edge960": "Longest edge 960 (upscaled to 1152)",
    "edge768": "**Longest edge 768**", "edge576": "Longest edge 576 (upscaled to 768)",
    "nosplit": "Single tile, no splitting",
    "keep0.5:uniform": "Prune to 50% per tile, uniform",
    "keep0.25:uniform": "Prune to 25% per tile, uniform",
    "keep0.25:random": "Prune to 25% per tile, random",
    "keep0.25:pool": "Prune to 25% per tile, 2-D mean pooling",
    "keep0.25:norm": "Prune to 25% per tile, largest norm",
    "keep0.077:uniform": "Prune to 7.7% per tile, uniform",
}


def load(results, name):
    return json.loads((results / f"{name}.json").read_text())


def n0(x):
    return f"{x:,.0f}"


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def pval(p):
    """A p-value as it reads after "p": "= 0.002", "< 0.001"."""
    return "< 0.0001" if p < 1e-4 else "< 0.001" if p < 1e-3 else f"= {p:.3f}"


def signed(x, digits=1):
    return f"{x:+.{digits}f}".replace("-", "−")


def differs(p, alpha=0.05):
    return "a real difference" if p < alpha else "not distinguishable"


def table(header, rows, align):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def paired_speedup_ci(records, key_base, key_other):
    """Median over samples of time(base) / time(other), with a bootstrap 95% interval."""
    per = {}
    for r in records:
        per.setdefault(r["sample_id"], {}).setdefault(r["config"], []).append(r["generate_ms"])
    ratios = [statistics.median(d[key_base]) / statistics.median(d[key_other])
              for d in per.values() if key_base in d and key_other in d]
    lo, hi = bootstrap_ci(ratios, stat=statistics.median, n_boot=4000)
    return statistics.median(ratios), lo, hi


def paired_delta_ci(records, key_a, key_b):
    """Bootstrap 95% interval of accuracy(B) - accuracy(A), resampling samples as pairs."""
    per = {}
    for r in records:
        per.setdefault(r["sample_id"], {}).setdefault(r["config"], []).append(r["correct"])
    diffs = [sum(d[key_b]) / len(d[key_b]) - sum(d[key_a]) / len(d[key_a])
             for d in per.values() if key_a in d and key_b in d]
    lo, hi = bootstrap_ci(diffs)
    return 100 * lo, 100 * hi


def with_metric(records, metric):
    """The same records, re-scored with another ChartQA metric (predictions are stored)."""
    return [dict(r, correct=score_chartqa(r["pred"], r["gold"], metric)) for r in records]


def delta_p(records, key_a, key_b):
    """Accuracy of B minus A in points, and the McNemar p-value, on the samples both share.

    The two configurations may come from runs of different length (bf16 on the whole
    split, nf4 on its first 300 questions); both numbers use only the shared samples.
    """
    ids = ({r["sample_id"] for r in records if r["config"] == key_a} &
           {r["sample_id"] for r in records if r["config"] == key_b})
    shared = [r for r in records if r["sample_id"] in ids]
    acc = lambda k: 100 * sum(r["correct"] for r in shared if r["config"] == k) / \
        sum(1 for r in shared if r["config"] == k)
    return acc(key_b) - acc(key_a), paired_accuracy(shared, key_a, key_b)["p_value"]


def median_ratio_ci(a, b, n_boot=4000, seed=0):
    """median(b) / median(a) for two independent samples, with a bootstrap 95% interval."""
    import random
    rng = random.Random(seed)
    boots = sorted(statistics.median(rng.choices(b, k=len(b))) /
                   statistics.median(rng.choices(a, k=len(a))) for _ in range(n_boot))
    return statistics.median(b) / statistics.median(a), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def values(results):
    v = {}
    # ---- gate 0: noise ------------------------------------------------------------
    g0 = load(results, "gate0_latency")
    cv = robust_cv(g0["raw_generate_ms"])
    v.update(noise_cv=pct(cv), noise_threshold=pct(3 * cv), gate0_runs=g0["generate_ms"]["n"],
             noise_plain_cv=pct(g0["generate_ms"]["cv_pct"]))

    # ---- where the time goes --------------------------------------------------------
    bd, pc = load(results, "breakdown"), load(results, "preprocess_cost")
    m, s = bd["median_ms"], bd["share_pct"]
    prep = pc["1536"]
    v.update(vision_pct=pct(s["vision"]), prefill_pct=pct(s["llm_prefill"]),
             answer_pct=pct(s["decode"]), answer_tokens=n0(bd["answer_tokens_median"]),
             image_tokens=n0(bd["image_tokens_median"]), input_tokens=n0(bd["input_tokens_median"]),
             tiles=n0(bd["image_tokens_median"] / TOKENS_PER_TILE),
             image_share=pct(100 * bd["image_tokens_median"] / bd["input_tokens_median"], 0),
             breakdown_samples=bd["detail"]["vision"]["n"],
             # Amdahl: pruning after the encoder can only shrink the language-model prefill
             prune_ceiling=f"{1 / (1 - s['llm_prefill'] / 100):.2f}×",
             encoder_side_pct=pct(s["vision"] + s["connector"] + s["llm_prefill"], 0))
    v["table_breakdown"] = table(
        ["Component", "Time", "Share"],
        [["**Vision encoder**", f"{m['vision']:.0f} ms", f"**{pct(s['vision'])}**"],
         ["Connector", f"{m['connector']:.1f} ms", pct(s["connector"])],
         ["Language-model prefill", f"{m['llm_prefill']:.0f} ms", pct(s["llm_prefill"])],
         [f"Generating the answer (median {bd['answer_tokens_median']:.0f} tokens, "
          f"at most {bd['max_new_tokens']})", f"{m['decode']:.0f} ms", pct(s["decode"])],
         ["*(CPU-side image preprocessing, before the GPU)*",
          f"*{prep['preprocess_ms']['median']:.0f} ms*", f"*{pct(prep['preprocess_share_pct'])}*"]],
        ["---", "---:", "---:"])

    # ---- lever sweep ---------------------------------------------------------------
    sw = load(results, "gate2_sweep")
    base = "baseline"
    rows, lever = [], {}
    others = [k for k in sw["configs"] if k != base]
    holm_p = dict(zip(others, holm([paired_accuracy(sw["records"], base, k)["p_value"] for k in others])))
    for key, c in sw["configs"].items():
        short = key.split("(")[0]
        recs = [r for r in sw["records"] if r["config"] == key]
        (lo, hi), n = accuracy_ci(recs)
        acc = 100 * c["accuracy"]
        if key == base:
            rows.append([LABELS[short], n0(c["image_tokens_median"]),
                         f"{acc:.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]", "—", "—", "—", "1.00×"])
            base_acc = acc
            v["sweep_baseline_acc"] = pct(acc)
            continue
        pa = paired_accuracy(sw["records"], base, key)
        d = acc - 100 * sw["configs"][base]["accuracy"]
        sp, sp_lo, sp_hi = paired_speedup_ci(sw["records"], base, key)
        lever[short] = dict(acc=acc, d=d, p=pa["p_value"], sp=sp, sp_lo=sp_lo,
                            tokens=c["image_tokens_median"])
        rows.append([LABELS[short], n0(c["image_tokens_median"]),
                     f"{acc:.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]", signed(d),
                     pval(pa["p_value"]).replace("= ", ""), pval(holm_p[key]).replace("= ", ""),
                     f"{sp:.2f}× [{sp_lo:.2f}–{sp_hi:.2f}]"])
    v["table_levers"] = table(
        ["Configuration", "Image tokens", "Accuracy [95% CI]", "Δ points", "p", "p (Holm)",
         "Speedup [95% CI]"],
        rows, ["---", "---:", "---", "---:", "---:", "---:", "---:"])
    v["sweep_comparisons"] = len(others)
    v["metric"] = sw.get("metric", "relaxed")
    v.update(sweep_samples=sw["samples"], sweep_rounds=sw["rounds"])
    for short, L in lever.items():
        tag = short.replace(":", "_").replace(".", "")
        v[f"{tag}_speedup"] = f"{L['sp']:.2f}×"
        v[f"{tag}_acc"] = pct(L["acc"])
        v[f"{tag}_delta"] = signed(L["d"])
        v[f"{tag}_p"] = pval(L["p"])
        v[f"{tag}_tokens"] = n0(L["tokens"])
    prune = [L["sp"] for k, L in lever.items() if k.startswith("keep")]
    v["prune_speedup_range"] = f"{min(prune):.2f}–{max(prune):.2f}×"
    # token selection methods against each other, at the same 25% budget
    k25 = lambda m: f"keep0.25:{m}(keep=0.25,{m})"
    for a, b in (("uniform", "norm"), ("random", "norm"), ("uniform", "random"), ("uniform", "pool")):
        pa = paired_accuracy(sw["records"], k25(a), k25(b))
        v[f"sel_{a}_vs_{b}"] = (f"{pa['only_b_correct']} vs {pa['only_a_correct']} discordant, "
                                f"p {pval(pa['p_value'])}")
    pa = paired_accuracy(sw["records"], "keep0.5:uniform(keep=0.5,uniform)", k25("norm"))
    v["sel_norm25_vs_uniform50_p"] = pval(pa["p_value"])
    pa = paired_accuracy(sw["records"], k25("norm"), "edge768(edge=768)")
    v["sel_norm25_vs_edge768_p"] = pval(pa["p_value"])
    v["norm25_holm_p"] = pval(holm_p[k25("norm")])

    # ---- the main lever, confirmed on more samples ---------------------------------------
    cf = load(results, "gate2_confirm")
    e768 = "edge768(edge=768)"
    pa = paired_accuracy(cf["records"], base, e768)
    lo, hi = paired_delta_ci(cf["records"], base, e768)
    sp, sp_lo, sp_hi = paired_speedup_ci(cf["records"], base, e768)
    rows = []
    for sub in ("human", "augmented"):
        recs = [r for r in cf["records"] if r.get("subset") == sub]
        if not recs:
            continue
        d, p = delta_p(recs, base, e768)
        acc = lambda k: 100 * sum(r["correct"] for r in recs if r["config"] == k) / sum(1 for r in recs if r["config"] == k)
        label = {"human": "Human-written questions", "augmented": "Generated questions"}[sub]
        rows.append([f"{label} ({len({r['sample_id'] for r in recs})})", pct(acc(base)), pct(acc(e768)),
                     f"{signed(d)} points, p {pval(p)}"])
    v["table_confirm_subsets"] = table(["", "Baseline", "Edge 768", "Difference"], rows,
                                       ["---", "---:", "---:", "---"]) if rows else ""
    v.update(confirm_samples=f"{cf['samples']:,}",
             confirm_base_acc=pct(100 * cf["configs"][base]["accuracy"]),
             confirm_e768_acc=pct(100 * cf["configs"][e768]["accuracy"]),
             confirm_e768_delta=signed(100 * (cf["configs"][e768]["accuracy"]
                                              - cf["configs"][base]["accuracy"])),
             confirm_e768_delta_ci=f"[{signed(lo)}, {signed(hi)}]",
             confirm_e768_p=pval(pa["p_value"]),
             confirm_e768_loss=f"{100 * (cf['configs'][base]['accuracy'] - cf['configs'][e768]['accuracy']):.1f}",
             confirm_e768_discordant=f"{pa['only_a_correct']} vs {pa['only_b_correct']}",
             confirm_e768_verdict=(
                 f"at a cost of {100 * (cf['configs'][base]['accuracy'] - cf['configs'][e768]['accuracy']):.1f} "
                 f"accuracy points (p {pval(pa['p_value'])}, {cf['samples']:,} questions)"
                 if pa["p_value"] < 0.05 else
                 f"with an accuracy change of {signed(100 * (cf['configs'][e768]['accuracy'] - cf['configs'][base]['accuracy']))} "
                 f"points that {cf['samples']:,} questions cannot tell apart from zero (p {pval(pa['p_value'])})"),
             confirm_e768_significant="a real trade-off" if pa["p_value"] < 0.05 else
                 "an accuracy cost too small to resolve even on the whole validation split",
             confirm_e768_speedup=f"{sp:.2f}×",
             confirm_e768_speed_ci=f"[{sp_lo:.2f}–{sp_hi:.2f}]")

    # Why the whole split: the power of McNemar's test for this loss, and what the first
    # questions of the same run show. Runs share their first questions (fixed shuffled
    # order), so the sweep's 100 and the nf4 run's 300 are prefixes of this run.
    discordant = pa["only_a_correct"] + pa["only_b_correct"]
    pi_d, pi_b = discordant / pa["n_pairs"], pa["only_a_correct"] / discordant
    first300 = {r["sample_id"] for r in load(results, "gate5_nf4")["records"]}
    prefixes = [("first 100 (those of the lever sweep)", {r["sample_id"] for r in sw["records"]}),
                ("first 300 (those of the nf4 run)", first300),
                ("the whole split", None)]
    rows = []
    for label, ids in prefixes:
        recs = cf["records"] if ids is None else [r for r in cf["records"] if r["sample_id"] in ids]
        n_q = len({r["sample_id"] for r in recs})
        d, p = delta_p(recs, base, e768)
        rows.append([f"{n_q:,}", label, f"{mcnemar_power(n_q, pi_d, pi_b):.2f}",
                     f"{signed(d)} points, p {pval(p)}"])
    v["confirm_power_first100"] = rows[0][2]
    v["table_confirm_power"] = table(
        ["Questions", "Which questions", "Power for this loss", "Measured on them"], rows,
        ["---:", "---", "---:", "---"])
    n80 = samples_for_power(pi_d, pi_b)
    v.update(confirm_pi_d=pct(100 * pi_d), confirm_pi_b=pct(100 * pi_b),
             confirm_discordant_total=f"{discordant:,}",
             confirm_n80=f"about {n80:,}" if n80 else "more than 100,000")

    # The earlier lenient metric on the same first 300 questions (the measurement-mistakes table)
    first = [r for r in cf["records"] if r["sample_id"] in first300]
    base_acc = lambda recs: 100 * sum(r["correct"] for r in recs if r["config"] == base) / \
        sum(1 for r in recs if r["config"] == base)
    relaxed300, lenient300 = with_metric(first, "relaxed"), with_metric(first, "lenient")
    d_len, p_len = delta_p(lenient300, base, e768)
    v.update(first300_relaxed_acc=pct(base_acc(relaxed300)), first300_lenient_acc=pct(base_acc(lenient300)),
             first300_lenient_gap=f"{base_acc(lenient300) - base_acc(relaxed300):.1f}",
             first300_lenient_e768_delta=signed(d_len), first300_lenient_e768_p=pval(p_len),
             first300_n=f"{len(first300):,}")

    # ---- quantization ----------------------------------------------------------------
    q = load(results, "gate3_quant")
    v["table_quant"] = table(
        ["Mode", "Latency (median)", "Peak VRAM", "Top-1 agreement", "Max logit difference"],
        [[f"{'**' + mode + '**' if mode == 'nf4' else mode}{' (reference)' if mode == 'bf16' else ''}",
          f"{q[mode]['latency_ms']['median']:,.0f} ms", f"{q[mode]['peak_vram_mb']:,.0f} MB",
          pct(100 * q[mode]["parity"]["top1_agreement"], 0),
          f"{q[mode]['parity']['max_abs_diff']:.2f}"] for mode in q],
        ["---", "---:", "---:", "---:", "---:"])
    v.update(quant_parity_samples=q["bf16"]["parity"]["samples"],
             quant_latency_n=q["bf16"]["latency_ms"]["n"],
             int8_slowdown=f"{q['int8']['latency_ms']['median'] / q['bf16']['latency_ms']['median']:.1f}×")

    nf = load(results, "gate5_nf4")
    cb, ce = compare_runs(cf, nf, base), compare_runs(cf, nf, e768)
    slower = lambda c: 100 * (1 / c["speed_b_over_a"] - 1)
    v["nf4_acc_verdict"] = ("a statistically significant loss" if cb["p_value"] < 0.05 else
                            "a loss that is not significant at the 0.05 level" if cb["p_value"] < 0.2 else
                            "not distinguishable from no change")
    v.update(nf4_samples=cb["n"], nf4_delta=signed(cb["acc_b"] - cb["acc_a"]),
             nf4_p=pval(cb["p_value"]), nf4_slower=pct(slower(cb), 0),
             nf4_slower_e768=pct(slower(ce), 0),
             nf4_vram_cut=pct(100 * (1 - cb["vram_b"] / cb["vram_a"]), 0))
    v["table_nf4"] = table(
        ["", "bf16", "nf4", "Difference"],
        [["Accuracy (baseline)", pct(cb["acc_a"]), pct(cb["acc_b"]),
          f"{signed(cb['acc_b'] - cb['acc_a'])} points, p {pval(cb['p_value'])}"],
         ["Accuracy (edge 768)", pct(ce["acc_a"]), pct(ce["acc_b"]),
          f"{signed(ce['acc_b'] - ce['acc_a'])} points, p {pval(ce['p_value'])}"],
         ["Speed, paired (baseline)", "1.00×", f"{cb['speed_b_over_a']:.2f}× [{cb['speed_ci95'][0]:.2f}–{cb['speed_ci95'][1]:.2f}]",
          f"**{pct(slower(cb), 0)} slower**"],
         ["Speed, paired (edge 768)", "1.00×", f"{ce['speed_b_over_a']:.2f}× [{ce['speed_ci95'][0]:.2f}–{ce['speed_ci95'][1]:.2f}]",
          f"**{pct(slower(ce), 0)} slower**"],
         ["Peak VRAM", f"{cb['vram_a']:,.0f} MB", f"**{cb['vram_b']:,.0f} MB**",
          f"**−{pct(100 * (1 - cb['vram_b'] / cb['vram_a']), 0)}**"]],
        ["---", "---:", "---:", "---"])

    # ---- which conclusions depend on the ChartQA metric ---------------------------------
    pr = load(results, "gate6_prompt")
    tag = lambda recs, t: [dict(r, config=f"{t}|{r['config']}") for r in recs]
    k25 = lambda m: f"keep0.25:{m}(keep=0.25,{m})"
    checks = [
        (f"Edge 768 vs baseline ({cf['samples']:,} questions)", cf["records"], base, e768),
        (f"Single tile vs baseline ({sw['samples']})", sw["records"], base, "nosplit(nosplit)"),
        (f"Prune 25% uniform vs baseline ({sw['samples']})", sw["records"], base, k25("uniform")),
        (f"Largest-norm vs uniform, 25% ({sw['samples']})", sw["records"], k25("uniform"), k25("norm")),
        (f"nf4 vs bf16 ({cb['n']})", tag(cf["records"], "bf16") + tag(nf["records"], "nf4"),
         f"bf16|{base}", f"nf4|{base}"),
        (f"English vs Vietnamese instruction ({pr['samples']})", pr["records"], base, "prompten(prompt=en)"),
    ]
    rows = []
    for name, recs, a_, b_ in checks:
        cells = [name]
        for metric in ("relaxed", "exact_years", "lenient"):
            d, p = delta_p(with_metric(recs, metric), a_, b_)
            cells.append(f"{signed(d)}, p {pval(p)}")
        rows.append(cells)
    v["table_sensitivity"] = table(
        ["Comparison", "Relaxed accuracy (primary)", "Years exact", "Earlier lenient metric"],
        rows, ["---", "---", "---", "---"])
    lenient = with_metric(cf["records"], "lenient")
    v["lenient_base_acc"] = pct(100 * sum(r["correct"] for r in lenient if r["config"] == base) /
                                sum(1 for r in lenient if r["config"] == base))

    # ---- serving --------------------------------------------------------------------------
    rows, serving = [], {}
    for edge in (1536, 768):
        sv = load(results, f"gate4_serving_{edge}")
        serving[edge] = sv
        for c, x in sv.items():
            e2e = x["end_to_end_ms"]
            bold = edge == 768 and c == "1"
            rows.append([f"**{edge}**" if bold else edge, c,
                         f"{'**' if bold else ''}{e2e['median']:,.0f} ms{'**' if bold else ''}",
                         f"{e2e['p95']:,.0f} ms", f"{x['server_ms']['median']:,.0f} ms",
                         f"{'**' if bold else ''}{x['throughput_rps']:.2f} req/s{'**' if bold else ''}"])
    v["table_serving"] = table(["Longest edge", "Concurrency", "Median", "p95", "Server compute",
                                "Throughput"], rows, ["---:", "---:", "---:", "---:", "---:", "---:"])
    v["serve_requests"] = serving[768]["1"]["end_to_end_ms"]["n"]
    v["serve_throughput_gain"] = (f"{serving[768]['1']['throughput_rps'] / serving[1536]['1']['throughput_rps']:.2f}×")

    rows, verdicts = [], []
    for edge in (768, 1536):
        nat, dk = serving[edge]["1"], load(results, f"gate4_docker_{edge}")["1"]
        r, lo, hi = median_ratio_ci(nat["raw_end_to_end_ms"], dk["raw_end_to_end_ms"])
        verdicts.append(lo <= 1 <= hi)
        rows.append([edge, f"{nat['end_to_end_ms']['median']:,.0f} ms · {nat['throughput_rps']:.2f} req/s",
                     f"{dk['end_to_end_ms']['median']:,.0f} ms · {dk['throughput_rps']:.2f} req/s",
                     f"{signed(100 * (r - 1), 0)}% [{signed(100 * (lo - 1), 0)}, {signed(100 * (hi - 1), 0)}]"])
    v["table_docker"] = table(["Longest edge", "Native", "In Docker", "Latency difference [95% CI]"],
                              rows, ["---:", "---:", "---:", "---:"])
    v["docker_verdict"] = ("Both intervals include zero, so Docker cannot be said to be slower." if all(verdicts)
                           else "At least one interval excludes zero: Docker is measurably different there.")

    # ---- DocVQA sanity check and instruction language -----------------------------------------
    dv, de = load(results, "sanity_docvqa"), load(results, "sanity_docvqa_en")
    rows = []
    for name, d in (("Vietnamese instruction", dv), ("English instruction", de)):
        lo, hi = d["anls_ci95"]
        inside = lo <= d["published_test_anls"] <= hi
        rows.append([f"This project, {name} ({d['n']} samples, validation split)",
                     f"**{d['anls_mean']:.1f}**", f"[{lo:.1f}, {hi:.1f}]",
                     "yes" if inside else "**no**"])
    rows.append(["Published by the SmolVLM authors (full test split)",
                 f"{dv['published_test_anls']:.1f}", "—", "—"])
    v["table_docvqa"] = table(["", "ANLS", "95% CI", "Published value inside the CI?"], rows,
                              ["---", "---:", "---:", "---"])
    v["docvqa_samples"] = dv["n"]
    # How much of the lost score comes from answers that are a shortened form of the
    # reference ("Pfizer" for "Pfizer Pharmaceuticals Group"), which the brief-answer
    # instruction could cause, rather than from values the model misread?
    norm = lambda t: " ".join(t.lower().split())
    def short(r):
        p = norm(clean_answer(r["pred"]))
        return r["anls"] < 1 and any(p and p in norm(g) and p != norm(g) for g in r["golds"])
    v["docvqa_lost_points"] = f"{sum(1 - r['anls'] for r in dv['rows']) / len(dv['rows']) * 100:.0f}"
    v["docvqa_short_points"] = f"{sum(1 - r['anls'] for r in dv['rows'] if short(r)) / len(dv['rows']) * 100:.0f}"
    lo, hi = dv["anls_ci95"]
    v["docvqa_vi"] = f"{dv['anls_mean']:.1f}"
    v["docvqa_en"] = f"{de['anls_mean']:.1f}"
    v["docvqa_published"] = f"{dv['published_test_anls']:.1f}"
    v["docvqa_verdict"] = (
        f"The published {dv['published_test_anls']:.1f} lies inside our 95% interval "
        f"[{lo:.1f}, {hi:.1f}], so this check finds no evidence of a systematic fault in "
        f"prompting, scoring or preprocessing" if lo <= dv["published_test_anls"] <= hi else
        f"The published {dv['published_test_anls']:.1f} lies outside our 95% interval "
        f"[{lo:.1f}, {hi:.1f}]: the gap of {dv['published_test_anls'] - dv['anls_mean']:.0f} points "
        f"is systematic, not sampling noise")
    ca = compare_anls(dv["rows"], de["rows"])
    pr = load(results, "gate6_prompt")
    pp = paired_accuracy(pr["records"], base, "prompten(prompt=en)")
    v["table_prompt"] = table(
        ["Dataset", "Vietnamese instruction", "English instruction", "Paired test"],
        [[f"ChartQA ({pr['samples']} samples)", pct(100 * pr["configs"][base]["accuracy"]),
          pct(100 * pr["configs"]["prompten(prompt=en)"]["accuracy"]),
          f"McNemar: {pp['only_a_correct']} vs {pp['only_b_correct']} discordant, p {pval(pp['p_value'])}"],
         [f"DocVQA ({ca['n']} samples)", f"{ca['mean_a']:.1f} ANLS", f"{ca['mean_b']:.1f} ANLS",
          f"sign test: {ca['b_better']} vs {ca['b_worse']}, p {pval(ca['p_value'])}; "
          f"difference {signed(ca['diff'])} [{signed(ca['diff_ci95'][0])}, {signed(ca['diff_ci95'][1])}]"]],
        ["---", "---:", "---:", "---"])
    v.update(prompt_chartqa_verdict=differs(pp["p_value"]), prompt_docvqa_verdict=differs(ca["p_value"]),
             prompt_chartqa_p=pval(pp["p_value"]), prompt_docvqa_p=pval(ca["p_value"]),
             prompt_docvqa_diff=signed(ca["diff"]),
             prompt_docvqa_ci=f"[{signed(ca['diff_ci95'][0])}, {signed(ca['diff_ci95'][1])}]")

    # ---- repository facts ----------------------------------------------------------------------
    v["n_tests"] = sum(len(re.findall(r"^def test_", p.read_text(), re.M))
                       for p in (ROOT / "tests").glob("test_*.py"))
    return {k: str(x) for k, x in v.items()}


def render(template, vals):
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)) - set(vals))
    if missing:
        raise KeyError(f"template uses unknown values: {missing}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--list", action="store_true", help="print every available value")
    a = ap.parse_args()
    vals = values(Path(a.results))
    if a.list:
        for k, x in sorted(vals.items()):
            print(f"{k:<28} {x if chr(10) not in x else '<table>'}")
        return
    text = render(TEMPLATE.read_text(), vals)
    if a.write:
        README.write_text(text)
        print(f"wrote {README}")
    elif README.read_text() != text:
        sys.exit("README.md is out of date with results/ — run: python -m bench.report --write")
    else:
        print("README.md matches results/")


if __name__ == "__main__":
    main()
