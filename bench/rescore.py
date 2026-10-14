"""Re-score recorded ChartQA runs with a different metric, without re-running the model.

Decoding is greedy, so a run's predictions are fixed; every record keeps its
prediction and gold answer, and scoring is a pure function of the two. Timings are
left untouched. Also fills in each record's question subset (human or augmented).

    python -m bench.rescore results/gate2_sweep.json results/gate6_prompt.json ...
"""
import argparse, json
from pathlib import Path

from bench import harness
from bench.metrics import accuracy_ci, score_chartqa


def rescore_run(run, metric="relaxed", subsets=None):
    """Return a copy of a harness run scored with `metric`; subsets maps sample_id -> subset."""
    run = json.loads(json.dumps(run))                       # deep copy
    for r in run["records"]:
        r["correct"] = score_chartqa(r["pred"], r["gold"], metric)
        if subsets is not None:
            r["subset"] = subsets.get(r["sample_id"], "")
    for key, c in run["configs"].items():
        recs = [r for r in run["records"] if r["config"] == key]
        (lo, hi), n = accuracy_ci(recs)
        c.update(accuracy=sum(r["correct"] for r in recs) / len(recs),
                 accuracy_ci95=[lo, hi], n_samples=n)
    run["metric"] = metric
    return run


def chartqa_subsets(revision):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/ChartQA", split="val", revision=revision)
    return {i: {0: "human", 1: "augmented"}[v] for i, v in enumerate(ds["human_or_machine"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default="relaxed", help="relaxed | exact_years | lenient")
    a = ap.parse_args()
    subsets = {}                        # per dataset commit: rescore with the data the run used
    for f in a.files:
        run = json.loads(Path(f).read_text())
        if run.get("dataset", "chartqa") != "chartqa":
            print(f"skip {f}: not a ChartQA run"); continue
        # runs recorded before revisions were pinned carry none; fall back to the current main
        rev = (run.get("identity", {}).get("dataset_revision")
               or harness.resolve_revision(harness.DATASETS["chartqa"][0], "main", "dataset"))
        if rev not in subsets:
            subsets[rev] = chartqa_subsets(rev)
        new = rescore_run(run, a.metric, subsets[rev])
        Path(f).write_text(json.dumps(new, indent=2, ensure_ascii=False))
        accs = {k.split("(")[0]: f"{100 * v['accuracy']:.1f}%" for k, v in new["configs"].items()}
        print(f"{f}: {a.metric} -> {accs}")


if __name__ == "__main__":
    main()
