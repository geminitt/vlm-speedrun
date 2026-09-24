"""Gate 3 — quantisation, with an equivalence check.

Principle: every "faster" variant must prove it produces EQUIVALENT OUTPUT to the
original before it is allowed to claim any speed.

Memory constraint: the card has only 6 GB, so two models must never sit in VRAM
at once. Reference logits are computed once and moved to CPU, then the bf16 model
is freed.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples
from bench.latency_probe import check_gpu_idle, summarize, timed


def load_quantized(model_id, mode, device="cuda"):
    """mode: bf16 (reference) | fp16 | int8 | nf4"""
    from transformers import AutoModelForImageTextToText
    if mode in ("bf16", "fp16"):
        dtype = torch.bfloat16 if mode == "bf16" else torch.float16
        return AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype).to(device).eval()
    from transformers import BitsAndBytesConfig
    if mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=cfg, device_map=device).eval()


@torch.no_grad()
def reference_logits(model, runner, samples, cfg):
    """Compute reference logits once, on every sample, and move them to CPU."""
    return [model(**runner.prepare(s, cfg), logits_to_keep=1).logits[0, -1].float().cpu()
            for s in samples]


@torch.no_grad()
def compare_to_reference(model, runner, samples, cfg, refs):
    """Compare the variant's logits against the stored reference logits.

    Three numbers, increasingly strict: mean absolute difference, worst-case
    difference, and top-1 agreement. The last one is what actually affects output.
    """
    max_abs, mean_abs, same_top1 = [], [], []
    for s, ref in zip(samples, refs):
        cur = model(**runner.prepare(s, cfg), logits_to_keep=1).logits[0, -1].float().cpu()
        d = (ref - cur).abs()
        max_abs.append(d.max().item())
        mean_abs.append(d.mean().item())
        same_top1.append(int(ref.argmax() == cur.argmax()))
    return {"max_abs_diff": max(max_abs),
            "mean_abs_diff": statistics.fmean(mean_abs),
            "top1_agreement": statistics.fmean(same_top1),
            "samples": len(same_top1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--modes", default="bf16,fp16,int8,nf4")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="results/gate3_quant.json")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    a = ap.parse_args()
    busy = None if a.allow_busy_gpu else check_gpu_idle()
    if busy:
        raise SystemExit(busy)

    runner = Runner(a.model)
    samples = load_samples(a.samples, seed=0)
    cfg = Config("baseline")
    res = {}

    def measure(model):
        lat = []
        with torch.no_grad():
            for s in samples[:3]:                       # warm up
                model.generate(**runner.prepare(s, cfg), max_new_tokens=8,
                               do_sample=False)
            for s in samples:
                inputs = runner.prepare(s, cfg)
                for _ in range(a.repeats):
                    ms, _ = timed(lambda: model.generate(
                        **inputs, max_new_tokens=32, do_sample=False))
                    lat.append(ms)
        return lat

    modes = a.modes.split(",")

    print("\n--- bf16 (reference) ---")
    torch.cuda.reset_peak_memory_stats()
    lat = measure(runner.model)
    refs = reference_logits(runner.model, runner, samples, cfg)
    res["bf16"] = {"latency_ms": summarize(lat),
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                   "parity": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                              "top1_agreement": 1.0, "samples": len(refs)}}
    print(f"median {res['bf16']['latency_ms']['median']:.0f} ms | "
          f"VRAM {res['bf16']['peak_vram_mb']:.0f} MB")

    runner.model = None                                  # free before loading the next variant
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    for mode in [m for m in modes if m != "bf16"]:
        print(f"\n--- {mode} ---")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = load_quantized(a.model, mode)
        runner.model = model
        lat = measure(model)
        parity = compare_to_reference(model, runner, samples, cfg, refs)
        res[mode] = {"latency_ms": summarize(lat),
                     "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                     "parity": parity}
        v = res[mode]
        print(f"median {v['latency_ms']['median']:.0f} ms | "
              f"VRAM {v['peak_vram_mb']:.0f} MB | "
              f"top-1 agreement {100*parity['top1_agreement']:.0f}% | "
              f"max logit diff {parity['max_abs_diff']:.3f}")
        del model
        runner.model = None
        torch.cuda.empty_cache()

    base = res["bf16"]["latency_ms"]["median"]
    for m, v in res.items():
        v["speedup_vs_bf16"] = base / v["latency_ms"]["median"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
