"""Gate 0 — measure baseline latency and the NOISE of the measurement itself.

The question is not "how fast is the model" but:
  if we measure the same thing N times on this laptop GPU, how far apart are the numbers?
If that spread exceeds the improvement we later want to demonstrate, every
conclusion in the project would be meaningless.
"""
import argparse, json, statistics, subprocess, threading, time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def make_image(size=512, seed=0):
    """Synthetic image: no network needed, fixed by seed."""
    g = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size, size, 3, generator=g) * 96 + 80).to(torch.uint8).numpy()
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for i in range(6):  # a few shapes so the image is not pure noise
        x, y = 40 + i * 70, 60 + (i % 3) * 120
        d.rectangle([x, y, x + 55, y + 85], fill=(30 + 30 * i, 200 - 20 * i, 120))
    return img


class GpuSampler(threading.Thread):
    """Sample clock, temperature and power while running."""
    Q = "clocks.current.graphics,temperature.gpu,power.draw,utilization.gpu"

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period, self.rows, self._ev = period, [], threading.Event()

    def run(self):
        while not self._ev.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.Q}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                clk, temp, pw, util = [p.strip() for p in out.split(",")]
                self.rows.append({"t": time.time(), "clock_mhz": float(clk),
                                  "temp_c": float(temp), "power_w": float(pw),
                                  "util_pct": float(util)})
            except Exception:
                pass
            self._ev.wait(self.period)

    def stop(self):
        self._ev.set()


def timed(fn):
    """Time with CUDA events, not the CPU clock."""
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out  # milliseconds


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    q1, med, q3 = xs[n // 4], statistics.median(xs), xs[(3 * n) // 4]
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"n": n, "min": xs[0], "p25": q1, "median": med, "p75": q3,
            "p95": xs[min(n - 1, int(0.95 * n))], "max": xs[-1],
            "mean": mean, "std": sd, "cv_pct": 100 * sd / mean if mean else 0.0,
            "iqr_pct": 100 * (q3 - q1) / med if med else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default="results/gate0_latency.json")
    a = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} | {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16).to(dev).eval()

    img = make_image(a.image_size)
    # The question text is a fixed experimental input ("Describe this image."): the
    # recorded gate-0 numbers were measured with it, so it is left untranslated.
    msg = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": "Mô tả ảnh này."}]}]
    text = proc.apply_chat_template(msg, add_generation_prompt=True)
    inputs = proc(text=text, images=[img], return_tensors="pt").to(dev)

    # Count image tokens — the central number of the whole project
    ids = inputs["input_ids"][0]
    img_tok_id = getattr(model.config, "image_token_id", None)
    n_img_tok = int((ids == img_tok_id).sum()) if img_tok_id is not None else -1
    print(f"input tokens: {ids.numel()} | image tokens: {n_img_tok}")

    gen = lambda: model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                 do_sample=False)
    fwd = lambda: model(**inputs)

    for _ in range(a.warmup):
        with torch.no_grad():
            gen()

    sampler = GpuSampler(); sampler.start()
    lat_gen, lat_fwd = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(a.runs):
            ms, _ = timed(gen);  lat_gen.append(ms)
            ms, _ = timed(fwd);  lat_fwd.append(ms)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.runs}  generate={lat_gen[-1]:.1f} ms")
    wall = time.time() - t0
    sampler.stop(); sampler.join(timeout=2)

    half = len(lat_gen) // 2
    res = {
        "model": a.model, "device": dev,
        "input_tokens": int(ids.numel()), "image_tokens": n_img_tok,
        "max_new_tokens": a.max_new_tokens, "wall_seconds": wall,
        "generate_ms": summarize(lat_gen),
        "prefill_ms": summarize(lat_fwd),
        "drift": {"first_half_median": statistics.median(lat_gen[:half]),
                  "second_half_median": statistics.median(lat_gen[half:])},
        "gpu": {k: summarize([r[k] for r in sampler.rows]) if sampler.rows else None
                for k in ("clock_mhz", "temp_c", "power_w")},
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0,
        "raw_generate_ms": lat_gen,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    g = res["generate_ms"]
    d = res["drift"]
    print(f"\n--- generate ({a.max_new_tokens} tokens) ---")
    print(f"median {g['median']:.1f} ms | IQR {g['iqr_pct']:.1f}% of median "
          f"| CV {g['cv_pct']:.1f}% | p95 {g['p95']:.1f} ms")
    print(f"first half {d['first_half_median']:.1f} ms -> second half "
          f"{d['second_half_median']:.1f} ms "
          f"({100*(d['second_half_median']/d['first_half_median']-1):+.1f}%)")
    print(f"peak VRAM {res['peak_vram_mb']:.0f} MB | saved {a.out}")


if __name__ == "__main__":
    main()
