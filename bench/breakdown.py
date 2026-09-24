"""Break latency down into: vision encoder, connector, language-model prefill,
and decoding.

The question: when a vision-language model runs on a small GPU, where does the
time actually go? Only after answering that do we know what to optimise.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples, PROMPT
from bench.latency_probe import check_gpu_idle, summarize, timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/breakdown.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    busy = None if a.allow_busy_gpu else check_gpu_idle()
    if busy:
        raise SystemExit(busy)

    r = Runner(a.model)
    model, inner = r.model, r.model.model
    samples = load_samples(a.samples, seed=0)

    parts = {"vision": [], "connector": [], "llm_prefill": [], "decode": [],
             "total_generate": []}
    n_tok, n_answer, ids_len = [], [], []

    with torch.no_grad():
        warm = Config("warmup", max_new_tokens=8)   # use the real Config class,
        for s in samples[:2]:                        # not a stand-in object
            r.run(s, warm)
        for s in samples:
            inputs = r.prepare(s)
            ids = inputs["input_ids"][0]
            for _ in range(a.repeats):
                t_vis, out = timed(lambda: inner.vision_model(
                    pixel_values=inputs["pixel_values"].flatten(0, 1).to(model.dtype)))
                vis = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                t_con, feats = timed(lambda: inner.connector(vis))
                # logits for the last position only, as generate() computes them;
                # logits for all 1,251 positions would inflate the prefill share
                t_pre, _ = timed(lambda: model(**inputs, logits_to_keep=1))
                t_gen, out = timed(lambda: model.generate(
                    **inputs, do_sample=False, max_new_tokens=a.max_new_tokens))
                n_answer.append(out.shape[1] - ids.numel())   # stops early at end-of-answer
                parts["vision"].append(t_vis)
                parts["connector"].append(t_con)
                parts["llm_prefill"].append(max(0.0, t_pre - t_vis - t_con))
                parts["decode"].append(max(0.0, t_gen - t_pre))
                parts["total_generate"].append(t_gen)
            n_tok.append(int((ids == r.img_token_id).sum()))
            ids_len.append(int(ids.numel()))

    med = {k: statistics.median(v) for k, v in parts.items()}
    total = med["total_generate"]
    res = {"model": a.model, "image_tokens_median": statistics.median(n_tok),
           "input_tokens_median": statistics.median(ids_len),
           "answer_tokens_median": statistics.median(n_answer),
           "max_new_tokens": a.max_new_tokens,
           "median_ms": med,
           "share_pct": {k: 100 * v / total for k, v in med.items() if k != "total_generate"},
           "detail": {k: summarize(v) for k, v in parts.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\nimage tokens: {res['image_tokens_median']:.0f} | "
          f"answer tokens: {res['answer_tokens_median']:.0f} | total generate {total:.0f} ms")
    for k in ("vision", "connector", "llm_prefill", "decode"):
        print(f"  {k:12s} {med[k]:7.1f} ms  ({res['share_pct'][k]:5.1f}%)")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
