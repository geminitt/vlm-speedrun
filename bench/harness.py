"""Main benchmark harness for vlm-speedrun.

One command, one headline metric, and six anti-noise rules enforced in code:
  1. interleave configurations instead of running them in blocks
  2. shuffle the order within each round
  3. report median and IQR, never mean +/- standard deviation
  4. set the threshold for claiming an improvement from measured noise
  5. record clock, temperature and power alongside every measurement
  6. re-measure the control configuration throughout the session to detect drift
"""
import argparse, json, random, statistics, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from bench.latency_probe import GpuSampler, summarize, timed
from bench.metrics import accuracy_ci, relaxed_match, anls
from bench.prune import build_inputs

# The instruction appended to every question. These strings are an experimental
# condition, not documentation: the Vietnamese one produced every recorded result,
# so it must stay byte-identical for those results to remain reproducible.
PROMPTS = {
    "vi": "{q}\nTrả lời ngắn gọn, chỉ đưa ra đáp án.",
    "en": "{q}\nAnswer briefly with the answer only.",
}
PROMPT = PROMPTS["vi"]   # kept as the default so earlier results stay reproducible


@dataclass
class Config:
    """One configuration to compare."""
    name: str
    keep_ratio: float = 1.0          # fraction of image tokens kept (after the connector)
    method: str = "uniform"          # token selection: uniform | pool | random | norm
    prompt: str = "vi"               # instruction language: vi | en
    split: bool = True               # whether the image is split into tiles
    max_edge: int = 1536             # longest edge when tiling (SmolVLM default)
    max_new_tokens: int = 32

    def key(self):
        bits = []
        if self.keep_ratio < 1.0:
            bits.append(f"keep={self.keep_ratio:g},{self.method}")
        if self.prompt != "vi":
            bits.append(f"prompt={self.prompt}")
        if not self.split:
            bits.append("nosplit")
        elif self.max_edge != 1536:
            bits.append(f"edge={self.max_edge}")
        return f"{self.name}({';'.join(bits)})" if bits else self.name

    def proc_key(self):
        """Two configurations share an image processor if they tile the same way."""
        return (self.split, self.max_edge)

    @staticmethod
    def parse(spec):
        """Syntax: baseline | keep0.5:uniform | nosplit | edge768 | nosplit+keep0.5

        Several levers combine with a plus sign.
        """
        cfg = Config(name=spec)
        for part in spec.split("+"):
            if part in ("baseline", "1", "1.0"):
                continue
            if part == "nosplit":
                cfg.split = False
            elif part.startswith("edge"):
                cfg.max_edge = int(part[4:])
            elif part.startswith("prompt"):
                cfg.prompt = part[6:].lstrip("=")
            elif part.startswith("keep"):
                keep, _, method = part.partition(":")
                cfg.keep_ratio = float(keep[4:])
                cfg.method = method or "uniform"
            else:
                raise ValueError(f"unknown configuration: {part!r}")
        return cfg


@dataclass
class Record:
    config: str
    sample_id: int
    round_idx: int
    correct: bool
    pred: str
    gold: str
    prefill_ms: float
    generate_ms: float
    image_tokens: int
    input_tokens: int
    clock_mhz: float = 0.0
    temp_c: float = 0.0


class Runner:
    def __init__(self, model_id, dtype="bfloat16", device="cuda", quant="bf16"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = device
        self.model_id = model_id
        self.quant = quant
        self._procs = {}
        self.proc = self.processor_for(Config("baseline"))
        if quant == "bf16":
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=getattr(torch, dtype)).to(device).eval()
        else:
            from bench.quantize import load_quantized
            self.model = load_quantized(model_id, quant, device)
        self.img_token_id = getattr(self.model.config, "image_token_id", None)

    def processor_for(self, cfg):
        from transformers import AutoProcessor
        k = cfg.proc_key()
        if k not in self._procs:
            kw = {"do_image_splitting": cfg.split}
            if cfg.split:
                kw["size"] = {"longest_edge": cfg.max_edge}
            self._procs[k] = AutoProcessor.from_pretrained(self.model_id, **kw)
        return self._procs[k]

    def prepare(self, sample, cfg=None):
        tmpl = PROMPTS.get(getattr(cfg, "prompt", "vi"), PROMPT) if cfg else PROMPT
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": tmpl.format(q=sample["query"])}]}]
        proc = self.processor_for(cfg) if cfg is not None else self.proc
        text = proc.apply_chat_template(msg, add_generation_prompt=True)
        return proc(text=text, images=[sample["image"]],
                    return_tensors="pt").to(self.device)

    @torch.no_grad()
    def infer(self, sample, cfg: Config):
        """Lean path for a real service: generation ONLY.

        Unlike run(), this does not execute a separate prefill pass. That pass
        exists only to break down timing during benchmarking; left in the server,
        it made every request do twice the work.
        """
        inputs = self.prepare(sample, cfg)
        ids = inputs["input_ids"][0]
        n_img = int((ids == self.img_token_id).sum()) if self.img_token_id else -1
        if cfg.keep_ratio >= 1.0:
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=cfg.max_new_tokens)
            new_tokens = out[0][ids.numel():]
        else:
            embeds, mask, n_img = build_inputs(self.model, inputs,
                                               cfg.keep_ratio, cfg.method)
            out = self.model.generate(inputs_embeds=embeds, attention_mask=mask,
                                      do_sample=False,
                                      max_new_tokens=cfg.max_new_tokens)
            new_tokens = out[0]
        pred = self.processor_for(cfg).decode(new_tokens, skip_special_tokens=True)
        return pred.strip(), n_img, int(ids.numel())

    @torch.no_grad()
    def run(self, sample, cfg: Config):
        """Run one sample and time it.

        IMPORTANT: the vision encoder must sit INSIDE the timed region on both
        paths. If only the pruning path calls it before the clock starts, one side
        still carries the load while the other has put it down, and the measured
        speedup is inflated.
        """
        inputs = self.prepare(sample, cfg)
        ids = inputs["input_ids"][0]
        n_img_full = int((ids == self.img_token_id).sum()) if self.img_token_id else -1
        state = {"n_img": n_img_full, "n_in": int(ids.numel())}

        def full(gen: bool):
            if cfg.keep_ratio >= 1.0:
                if gen:
                    return self.model.generate(**inputs, do_sample=False,
                                               max_new_tokens=cfg.max_new_tokens)
                return self.model(**inputs)
            embeds, mask, n_kept = build_inputs(self.model, inputs,
                                                cfg.keep_ratio, cfg.method)
            state["n_img"], state["n_in"] = n_kept, int(embeds.shape[1])
            kw = {"inputs_embeds": embeds, "attention_mask": mask}
            if gen:
                return self.model.generate(**kw, do_sample=False,
                                           max_new_tokens=cfg.max_new_tokens)
            return self.model(**kw)

        prefill_ms, _ = timed(lambda: full(False))
        gen_ms, out = timed(lambda: full(True))
        new_tokens = out[0][ids.numel():] if cfg.keep_ratio >= 1.0 else out[0]
        pred = self.processor_for(cfg).decode(new_tokens, skip_special_tokens=True).strip()
        return pred, prefill_ms, gen_ms, state["n_img"], state["n_in"]


DATASETS = {
    # name -> (hub path, split, image column, question column, answer column)
    "chartqa": ("HuggingFaceM4/ChartQA", "val", "image", "query", "label"),
    "docvqa": ("lmms-lab/DocVQA", "validation", "image", "question", "answers"),
}


def load_samples(n, seed=0, dataset="chartqa"):
    """Take n samples, fixed by seed. The same seed always yields the same set."""
    from datasets import load_dataset
    path, split, c_img, c_q, c_a = DATASETS[dataset]
    kw = {"name": "DocVQA"} if dataset == "docvqa" else {}
    ds = load_dataset(path, split=split, **kw)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        row = ds[i]
        golds = row[c_a]
        out.append({"sample_id": i, "image": row[c_img], "query": row[c_q],
                    "gold": golds[0] if isinstance(golds, list) else golds,
                    "golds": golds if isinstance(golds, list) else [golds]})
    return out


def build_plan(configs, n_samples, rounds, seed):
    """Rules 1 and 2: every round runs ALL samples for EVERY configuration, shuffled.

    Rounds are therefore true replicates over the same sample set, which means:
      - comparisons between configurations are paired, on the same sample
      - drift over time is meaningful, since the first and last rounds share samples
    If each round used a different group of samples, most of the spread would come
    from images being larger or smaller, not from the configuration.
    """
    rng = random.Random(seed)
    plan = []
    for r in range(rounds):
        block = [(cfg, i) for cfg in configs for i in range(n_samples)]
        rng.shuffle(block)
        plan.extend((cfg, i, r) for cfg, i in block)
    return plan


def paired_speedup(records, base_key, other_key):
    """Paired comparison: for each sample, take the ratio of the two latencies.

    Returns the median ratio and its interquartile range. This removes the effect
    of some samples being heavier than others, which dominates the spread.
    """
    from collections import defaultdict
    per = defaultdict(dict)
    for r in records:
        per[r.sample_id].setdefault(r.config, []).append(r.generate_ms)
    ratios = []
    for sid, by_cfg in per.items():
        if base_key in by_cfg and other_key in by_cfg:
            b = statistics.median(by_cfg[base_key])
            o = statistics.median(by_cfg[other_key])
            if o > 0:
                ratios.append(b / o)  # >1 means the other configuration is faster
    if not ratios:
        return None
    ratios.sort()
    n = len(ratios)
    return {"n_pairs": n, "median_speedup": statistics.median(ratios),
            "p25": ratios[n // 4], "p75": ratios[(3 * n) // 4],
            "min": ratios[0], "max": ratios[-1]}


def summarize_config(records, noise_cv_pct):
    gen = [r.generate_ms for r in records]
    pre = [r.prefill_ms for r in records]
    k = sum(r.correct for r in records)
    n = len(records)
    (lo, hi), n_samples = accuracy_ci(records)   # one observation per sample, not per round
    return {
        "n": n,
        "n_samples": n_samples,
        "accuracy": k / n if n else 0.0,
        "accuracy_ci95": [lo, hi],
        "generate_ms": summarize(gen),
        "prefill_ms": summarize(pre),
        "prefill_share_pct": 100 * statistics.median(pre) / statistics.median(gen),
        "image_tokens_median": statistics.median([r.image_tokens for r in records]),
        "clock_mhz_median": statistics.median([r.clock_mhz for r in records]),
        "temp_c_median": statistics.median([r.temp_c for r in records]),
        "noise_threshold_pct": 3 * noise_cv_pct,  # rule 4
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-Instruct",
                    help="defaults to the project's main model; the 256M variant is for quick checks only")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--noise-cv", type=float, default=8.5,
                    help="noise measured in gate 0, in %%")
    ap.add_argument("--quant", default="bf16", help="bf16 | fp16 | int8 | nf4")
    ap.add_argument("--dataset", default="chartqa", help="chartqa | docvqa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gate1.json")
    a = ap.parse_args()

    configs = [Config.parse(c) for c in a.configs.split(",")]
    samples = load_samples(a.samples, a.seed, a.dataset)
    score = ((lambda p, s: relaxed_match(p, s['gold'])) if a.dataset == 'chartqa'
             else (lambda p, s: anls(p, s['golds']) >= 0.5))
    runner = Runner(a.model, quant=a.quant)

    print(f"model: {a.model} [{a.quant}] | {len(samples)} samples | {len(configs)} configs "
          f"| {a.rounds} rounds")

    # warm up so the first measurement does not count
    for s in samples[:3]:
        runner.run(s, configs[0])

    sampler = GpuSampler(); sampler.start()
    plan = build_plan(configs, len(samples), a.rounds, a.seed)
    records, t0 = [], time.time()
    for step, (cfg, i, r) in enumerate(plan, 1):
        s = samples[i]
        pred, pre_ms, gen_ms, n_img, n_in = runner.run(s, cfg)
        tele = sampler.rows[-1] if sampler.rows else {"clock_mhz": 0, "temp_c": 0}
        records.append(Record(
            config=cfg.key(), sample_id=s["sample_id"], round_idx=r,
            correct=score(pred, s), pred=pred, gold=s["gold"],
            prefill_ms=pre_ms, generate_ms=gen_ms,
            image_tokens=n_img, input_tokens=n_in,
            clock_mhz=tele["clock_mhz"], temp_c=tele["temp_c"]))
        if step % 25 == 0:
            done = sum(x.correct for x in records)
            print(f"  {step}/{len(plan)} | correct {done}/{len(records)} "
                  f"| {records[-1].generate_ms:.0f} ms | {tele['temp_c']:.0f}°C")
    sampler.stop(); sampler.join(timeout=2)

    by_cfg = {}
    for cfg in configs:
        rs = [r for r in records if r.config == cfg.key()]
        if rs:
            by_cfg[cfg.key()] = summarize_config(rs, a.noise_cv)

    # rule 6: check drift between the first and last round of the control configuration
    ctrl = configs[0].key()
    f = {r.sample_id: r.generate_ms for r in records
         if r.config == ctrl and r.round_idx == 0}
    l = {r.sample_id: r.generate_ms for r in records
         if r.config == ctrl and r.round_idx == a.rounds - 1}
    common = sorted(set(f) & set(l))
    drift = (100 * (statistics.median([l[i] / f[i] for i in common]) - 1)
             if common else None)

    speedups = {}
    if len(configs) > 1:
        base = configs[0].key()
        for cfg in configs[1:]:
            sp = paired_speedup(records, base, cfg.key())
            if sp:
                speedups[cfg.key()] = sp

    out = {"model": a.model, "quant": a.quant, "dataset": a.dataset,
           "samples": len(samples), "rounds": a.rounds,
           "paired_speedup_vs_baseline": speedups,
           "wall_seconds": time.time() - t0, "configs": by_cfg,
           "control_drift_pct": drift,
           "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
           "records": [asdict(r) for r in records]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== RESULTS ===")
    for k, v in by_cfg.items():
        ci = v["accuracy_ci95"]
        print(f"{k}: accuracy {100*v['accuracy']:.1f}% "
              f"[{100*ci[0]:.1f}–{100*ci[1]:.1f}] | "
              f"generate median {v['generate_ms']['median']:.0f} ms "
              f"(IQR {v['generate_ms']['iqr_pct']:.1f}%) | "
              f"prefill share {v['prefill_share_pct']:.0f}% | "
              f"image tokens {v['image_tokens_median']:.0f}")
    if speedups:
        print("\n--- speedup vs baseline (paired on the same samples) ---")
        thr = 1 + 3 * a.noise_cv / 100
        for k, v in speedups.items():
            ok = "✓" if v["median_speedup"] >= thr else "✗ below noise threshold"
            print(f"{k}: {v['median_speedup']:.2f}x "
                  f"[{v['p25']:.2f}–{v['p75']:.2f}] {ok}")
    print(f"control drift: {drift:+.1f}%" if drift is not None else "")
    print(f"peak VRAM {out['peak_vram_mb']:.0f} MB | saved {a.out}")


if __name__ == "__main__":
    main()
