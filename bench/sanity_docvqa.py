"""Sanity check: run on DocVQA and compare against the published number.

The SmolVLM authors report DocVQA (test) = 81.6. If our pipeline lands far from
that, something systematic is wrong — prompting, scoring or preprocessing — and
every other result in the project becomes suspect with it.

Note: we run on the validation split (test labels are not public) with the exact
ANLS metric, but on fewer samples, so only order-of-magnitude agreement is expected.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples
from bench.metrics import anls

PUBLISHED = 81.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="vi", help="vi | en")
    ap.add_argument("--out", default="results/sanity_docvqa.json")
    a = ap.parse_args()

    runner = Runner(a.model)
    samples = load_samples(a.samples, seed=0, dataset="docvqa")
    cfg = Config("baseline", max_new_tokens=a.max_new_tokens, prompt=a.prompt)

    scores, rows = [], []
    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            pred, n_img, _ = runner.infer(s, cfg)
            sc = anls(pred, s["golds"])
            scores.append(sc)
            rows.append({"sample_id": s["sample_id"], "pred": pred,
                         "golds": s["golds"], "anls": sc})
            if i % 25 == 0:
                print(f"  {i}/{len(samples)} | running ANLS "
                      f"{100*statistics.fmean(scores):.1f}")

    mean = 100 * statistics.fmean(scores)
    res = {"model": a.model, "dataset": "docvqa/validation", "n": len(scores),
           "anls_mean": mean, "published_test_anls": PUBLISHED,
           "gap": mean - PUBLISHED, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nour ANLS    : {mean:.1f}  ({len(scores)} samples, validation split)")
    print(f"published   : {PUBLISHED:.1f}  (full test split)")
    print(f"gap         : {mean - PUBLISHED:+.1f} points")
    print("verdict     : " + ("pipeline agrees with the published number, no systematic fault"
                              if abs(mean - PUBLISHED) < 10 else
                              "LARGE GAP — review prompting, scoring and preprocessing"))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
