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
from bench.metrics import accuracy_ci, bootstrap_ci, clean_answer, paired_accuracy, robust_cv

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE, README = ROOT / "README.template.md", ROOT / "README.md"

LABELS = {   # configuration key prefix -> README label
    "baseline": "Baseline (13 tiles, longest edge 1536)",
    "edge1152": "Longest edge 1152", "edge960": "Longest edge 960",
    "edge768": "**Longest edge 768**", "edge576": "Longest edge 576",
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
    for key, c in sw["configs"].items():
        short = key.split("(")[0]
        recs = [r for r in sw["records"] if r["config"] == key]
        (lo, hi), n = accuracy_ci(recs)
        acc = 100 * c["accuracy"]
        if key == base:
            rows.append([LABELS[short], n0(c["image_tokens_median"]),
                         f"{acc:.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]", "—", "—", "1.00×"])
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
                     pval(pa["p_value"]).replace("= ", ""), f"{sp:.2f}× [{sp_lo:.2f}–{sp_hi:.2f}]"])
    v["table_levers"] = table(
        ["Configuration", "Image tokens", "Accuracy [95% CI]", "Δ points", "p",
         "Speedup [95% CI]"],
        rows, ["---", "---:", "---", "---:", "---:", "---:"])
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

    # ---- the main lever, confirmed on more samples ---------------------------------------
    cf = load(results, "gate2_confirm")
    e768 = "edge768(edge=768)"
    pa = paired_accuracy(cf["records"], base, e768)
    lo, hi = paired_delta_ci(cf["records"], base, e768)
    sp, sp_lo, sp_hi = paired_speedup_ci(cf["records"], base, e768)
    v.update(confirm_samples=cf["samples"],
             confirm_base_acc=pct(100 * cf["configs"][base]["accuracy"]),
             confirm_e768_acc=pct(100 * cf["configs"][e768]["accuracy"]),
             confirm_e768_delta=signed(100 * (cf["configs"][e768]["accuracy"]
                                              - cf["configs"][base]["accuracy"])),
             confirm_e768_delta_ci=f"[{signed(lo)}, {signed(hi)}]",
             confirm_e768_p=pval(pa["p_value"]),
             confirm_e768_loss=f"{100 * (cf['configs'][base]['accuracy'] - cf['configs'][e768]['accuracy']):.1f}",
             confirm_e768_discordant=f"{pa['only_a_correct']} vs {pa['only_b_correct']}",
             confirm_e768_speedup=f"{sp:.2f}×",
             confirm_e768_speed_ci=f"[{sp_lo:.2f}–{sp_hi:.2f}]")

    # ---- quantisation ----------------------------------------------------------------
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
                            "not significant at the 0.05 level" if cb["p_value"] < 0.2 else
                            "no detectable loss")
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
         ["Speed, paired (baseline)", "1.00×", f"{cb['speed_b_over_a']:.2f}×",
          f"**{pct(slower(cb), 0)} slower**"],
         ["Speed, paired (edge 768)", "1.00×", f"{ce['speed_b_over_a']:.2f}×",
          f"**{pct(slower(ce), 0)} slower**"],
         ["Peak VRAM", f"{cb['vram_a']:,.0f} MB", f"**{cb['vram_b']:,.0f} MB**",
          f"**−{pct(100 * (1 - cb['vram_b'] / cb['vram_a']), 0)}**"]],
        ["---", "---:", "---:", "---"])

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

    rows, worst = [], 0.0
    for edge in (768, 1536):
        nat, dk = serving[edge]["1"], load(results, f"gate4_docker_{edge}")["1"]
        diff = 100 * (dk["end_to_end_ms"]["median"] / nat["end_to_end_ms"]["median"] - 1)
        worst = max(worst, abs(diff))
        rows.append([edge, f"{nat['end_to_end_ms']['median']:,.0f} ms · {nat['throughput_rps']:.2f} req/s",
                     f"{dk['end_to_end_ms']['median']:,.0f} ms · {dk['throughput_rps']:.2f} req/s",
                     f"{signed(diff, 0)}%"])
    v["table_docker"] = table(["Longest edge", "Native", "In Docker", "Difference"], rows,
                              ["---:", "---:", "---:", "---:"])
    v["docker_verdict"] = (f"Both differences are below the {pct(3 * cv)} noise threshold, so Docker "
                           f"cannot be said to be slower." if worst < 3 * cv else
                           f"The larger difference exceeds the {pct(3 * cv)} noise threshold: "
                           f"Docker is measurably slower here.")

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
