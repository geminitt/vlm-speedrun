"""Cổng 3 — lượng tử hoá, kèm kiểm tra tương đương.

Nguyên tắc: mọi phiên bản "nhanh hơn" phải chứng minh cho ra KẾT QUẢ TƯƠNG ĐƯƠNG
bản gốc trước khi được phép khoe tốc độ.

Ràng buộc bộ nhớ: card chỉ có 6 GB, nên không bao giờ được để hai mô hình cùng
nằm trong VRAM. Logit chuẩn được tính một lần rồi cất sang CPU, sau đó bản bf16
được giải phóng.
"""
import argparse, json, statistics
from pathlib import Path

import torch

from bench.harness import Runner, Config, load_samples
from bench.latency_probe import summarize, timed


def load_quantized(model_id, mode, device="cuda"):
    """mode: bf16 (gốc) | fp16 | int8 | nf4"""
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
        raise ValueError(f"chế độ lạ: {mode}")
    return AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=cfg, device_map=device).eval()


@torch.no_grad()
def reference_logits(model, runner, samples, cfg, n=8):
    """Tính logit chuẩn một lần rồi cất sang CPU."""
    return [model(**runner.prepare(s, cfg)).logits[0, -1].float().cpu()
            for s in samples[:n]]


@torch.no_grad()
def compare_to_reference(model, runner, samples, cfg, refs):
    """So logit của phiên bản đang xét với logit chuẩn đã cất sẵn.

    Ba con số, chặt dần: sai lệch trung bình, sai lệch lớn nhất, và tỉ lệ token
    đầu tiên được chọn trùng nhau. Con số cuối mới là thứ ảnh hưởng tới đầu ra.
    """
    max_abs, mean_abs, same_top1 = [], [], []
    for s, ref in zip(samples, refs):
        cur = model(**runner.prepare(s, cfg)).logits[0, -1].float().cpu()
        d = (ref - cur).abs()
        max_abs.append(d.max().item())
        mean_abs.append(d.mean().item())
        same_top1.append(int(ref.argmax() == cur.argmax()))
    return {"max_abs_diff": max(max_abs),
            "mean_abs_diff": statistics.fmean(mean_abs),
            "top1_agreement": statistics.fmean(same_top1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct")
    ap.add_argument("--modes", default="bf16,fp16,int8,nf4")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="results/gate3_quant.json")
    a = ap.parse_args()

    runner = Runner(a.model)
    samples = load_samples(a.samples, seed=0)
    cfg = Config("baseline")
    res = {}

    def measure(model):
        lat = []
        with torch.no_grad():
            for s in samples[:3]:                       # làm nóng
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

    print("\n--- bf16 (ban chuan) ---")
    torch.cuda.reset_peak_memory_stats()
    lat = measure(runner.model)
    refs = reference_logits(runner.model, runner, samples, cfg)
    res["bf16"] = {"latency_ms": summarize(lat),
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                   "parity": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                              "top1_agreement": 1.0}}
    print(f"trung vi {res['bf16']['latency_ms']['median']:.0f} ms | "
          f"VRAM {res['bf16']['peak_vram_mb']:.0f} MB")

    runner.model = None                                  # giải phóng trước khi nạp bản khác
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
        print(f"trung vi {v['latency_ms']['median']:.0f} ms | "
              f"VRAM {v['peak_vram_mb']:.0f} MB | "
              f"khop token dau {100*parity['top1_agreement']:.0f}% | "
              f"lech logit toi da {parity['max_abs_diff']:.3f}")
        del model
        runner.model = None
        torch.cuda.empty_cache()

    base = res["bf16"]["latency_ms"]["median"]
    for m, v in res.items():
        v["speedup_vs_bf16"] = base / v["latency_ms"]["median"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nda luu {a.out}")


if __name__ == "__main__":
    main()
