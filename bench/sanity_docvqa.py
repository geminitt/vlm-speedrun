"""Sanity check: run on DocVQA and compare against the published number.

The SmolVLM authors report DocVQA (test) = 81.6. A large gap is a reason to look
for a fault in prompting, scoring or preprocessing, but not proof of one.

Note: we run on the validation split (test labels are not public) with the exact
ANLS metric, on a sample, and with our own prompt and image settings. The verdict
therefore asks only whether the published number lies inside the bootstrap 95%
interval of our estimate — a fixed tolerance would say nothing about how much a
sample of this size can resolve. Because splits and setups differ, the check
cannot say where a gap comes from.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, add_revision_args, run_identity, Config, load_samples
from bench.metrics import anls, bootstrap_ci, clean_answer

PUBLISHED = 81.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="vi", help="vi | en")
    ap.add_argument("--out", default="results/sanity_docvqa.json")
    add_revision_args(ap)
    a = ap.parse_args()

    ident = run_identity(a, a.model, "docvqa")
    runner = Runner(a.model, revision=ident["model_revision"])
    samples = load_samples(a.samples, seed=0, dataset="docvqa", revision=ident["dataset_revision"])
    cfg = Config("baseline", max_new_tokens=a.max_new_tokens, prompt=a.prompt)

    scores, rows = [], []
    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            pred, n_img, _ = runner.infer(s, cfg)
            sc = anls(clean_answer(pred), s["golds"])
            scores.append(sc)
            rows.append({"sample_id": s["sample_id"], "pred": pred,
                         "golds": s["golds"], "anls": sc})
            if i % 25 == 0:
                print(f"  {i}/{len(samples)} | running ANLS "
                      f"{100*statistics.fmean(scores):.1f}")

    mean = 100 * statistics.fmean(scores)
    lo, hi = (100 * v for v in bootstrap_ci(scores))
    res = {"model": a.model, "identity": ident, "dataset": "docvqa/validation", "prompt": a.prompt,
           "n": len(scores), "anls_mean": mean, "anls_ci95": [lo, hi],
           "published_test_anls": PUBLISHED, "gap": mean - PUBLISHED, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nour ANLS    : {mean:.1f}  95% CI [{lo:.1f}, {hi:.1f}]  ({len(scores)} samples, validation split)")
    print(f"published   : {PUBLISHED:.1f}  (full test split)")
    print(f"gap         : {mean - PUBLISHED:+.1f} points")
    print("verdict     : " + ("published number inside our 95% interval: no gap beyond sampling error"
                              if lo <= PUBLISHED <= hi else
                              "published number OUTSIDE our 95% interval: a gap beyond sampling error")
          + " (different splits and setups: this check does not attribute a cause)")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
