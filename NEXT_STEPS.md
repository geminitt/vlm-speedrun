# Project status

Updated: 2026-09-25. **Every gate is complete and reproducible with `./speedrun.sh`.**
The numbers live in [README.md](README.md), which `bench/report.py` generates from
`results/`; this file deliberately repeats none of them, so it cannot drift from the data.

| Gate | Scope | Where the result is |
|---|---|---|
| 0 | Noise of the measurement itself | README, rule 5 · `results/gate0_latency.json` |
| 1 | Where the time goes | README, "Where the time goes" · `results/breakdown.json` |
| 2 | Levers that reduce image tokens | README, lever table and figure · `results/gate2_sweep.json`, `gate2_confirm.json` |
| 3 | Quantisation | README, "Quantisation" · `results/gate3_quant.json`, `gate5_nf4.json` |
| 4 | gRPC server, native and in Docker | README, "Serving over gRPC" · `results/gate4_*.json` |
| 5 | DocVQA check and instruction language | README, last two result sections · `results/sanity_docvqa*.json`, `gate6_prompt.json` |

## Audits of 2026-09-25

Two full reviews of the repository found errors in the method, the code and the README.
All are fixed and every result was re-measured; the README table of measurement mistakes
lists each one. The changes that altered conclusions:

- Token pruning no longer deletes the tile-layout tokens, and the fourth selection
  method, never run before, was run: **which tokens are kept does matter** —
  largest-norm selection clearly beats evenly spaced, random and pooled selection.
- The single-tile configuration is about three times faster than the baseline; the
  earlier, slower number came from a run that shared the GPU with another process.
- ChartQA is now scored with the benchmark's relaxed accuracy instead of a lenient
  home-made metric. Edge 768's accuracy cost shrank from 7 points to about 2; it is
  resolved (p < 0.05) only on the whole validation split of 1,920 questions, which the
  main comparison now uses. The README shows every main comparison under three metrics.
- With the official ANLS and 300 questions, the DocVQA score sits clearly below the
  published one; the check now reports that gap instead of passing.
- Differences are judged by their 95% interval, with Holm's correction for the sweep.

Guards added so these cannot recur: timing scripts refuse a busy GPU; the harness fails
a run with foreign GPU memory or many timing spikes, and checkpoints every measurement
so a run interrupted by a shutdown resumes; pipeline checks write to `results/fast/`;
and CI fails if README.md and `results/` disagree.

## Optional extensions

Not required; ordered by how worthwhile they are:

1. **Activation quantisation for the vision encoder** (W8A8 or FP8, which this Ada GPU
   supports) — the encoder is the compute-bound majority of the time, the only place
   quantisation could buy speed in this workload.
2. **Attention-score token pruning (FastV-style)** — largest-norm selection already
   shows that informed selection matters; attention scores are the next signal to try.
   Still capped by Amdahl's law, since pruning cannot touch the encoder.
3. **Explain the DocVQA gap** — rerun with the authors' evaluation prompt and image
   settings to see how much of it is setup rather than the model.
4. **CUDA graphs or a fused 4-bit kernel** — to turn nf4's smaller weights into faster
   decoding by removing the CPU-side work that sets the pace of each step. A timeline
   profile (Nsight Systems) would first show whether that work is kernel launches or
   synchronisation.
5. **Server-side batching** — the server handles one request at a time; batching
   raises throughput where decoding dominates, and nf4's freed memory makes room for it.
6. **A second model** (e.g. Qwen2.5-VL-3B) — to test whether "the vision encoder takes
   over half the time" holds beyond SmolVLM.

## Notes for rerunning

- The GPU must be idle; timing scripts check this and stop otherwise
  (`--allow-busy-gpu` overrides, and the harness then flags the run).
- `START=n ./speedrun.sh` resumes from step n after a failure (`END=n` stops after step
  n); a harness run interrupted midway resumes from its checkpoint when rerun. `FAST=1`
  checks the pipeline in about 15 minutes without touching reference results.
- Launch long runs detached so they survive the terminal closing:
  `setsid nohup ./speedrun.sh > results/run.log 2>&1 &`
- Edit `README.template.md`, never `README.md`, then run `python -m bench.report --write`.
- Do not use `pkill -f` with a pattern that also matches the command being typed; it
  kills its own shell. Use a bracket pattern such as `"bench\.qu[a]ntize"`.
