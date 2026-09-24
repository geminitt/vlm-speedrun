"""Measure CPU-side image preprocessing — the part left out of every earlier benchmark.

The gate 1 and 2 harness calls the processor BEFORE the clock starts, so the cost
of tiling, resizing and normalising the image never appears in any number. A real
service pays for it anyway.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples
from bench.latency_probe import summarize, timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--edges", default="1536,768")
    ap.add_argument("--out", default="results/preprocess_cost.json")
    a = ap.parse_args()

    runner = Runner(a.model)
    samples = load_samples(a.samples, seed=0)
    res = {}

    for edge in [int(e) for e in a.edges.split(",")]:
        cfg = Config("x", max_edge=edge)
        prep, gpu = [], []
        for s in samples[:2]:
            runner.run(s, cfg)                       # warm up
        for s in samples:
            t0 = time.perf_counter()
            inputs = runner.prepare(s, cfg)          # CPU: resize, tile, normalise
            prep.append((time.perf_counter() - t0) * 1000)
            with torch.no_grad():
                ms, _ = timed(lambda: runner.model.generate(
                    **inputs, max_new_tokens=32, do_sample=False))
            gpu.append(ms)
        total = statistics.median(prep) + statistics.median(gpu)
        res[str(edge)] = {"preprocess_ms": summarize(prep), "gpu_ms": summarize(gpu),
                          "total_ms": total,
                          "preprocess_share_pct": 100 * statistics.median(prep) / total}
        v = res[str(edge)]
        print(f"edge {edge}: preprocess {v['preprocess_ms']['median']:.0f} ms "
              f"| GPU {v['gpu_ms']['median']:.0f} ms "
              f"| preprocess is {v['preprocess_share_pct']:.0f}% of {total:.0f} ms total")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
