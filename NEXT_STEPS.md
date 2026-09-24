# Project status

Updated: 2026-09-24. **All six gates are complete.**

| Gate | Scope | Result |
|---|---|---|
| 0 | Noise of the measurement itself | noise 4.4% (CV, 2.2B model), improvement-claim threshold 13.2% |
| 1 | Accuracy + latency harness | baseline 64.7% ChartQA / 990 ms |
| 2 | Levers that reduce image tokens | longest edge 1536→768: **1.93×**, −7.3 points (p = 0.002) |
| 3 | Quantisation | nf4 cuts **55% VRAM**, 8% slower, signs of −4 points (p = 0.081) |
| 4 | gRPC server | edge 768 gives **2.02 requests/s**, double the baseline |
| 5 | Sanity check + prompt ablation | DocVQA ANLS 73.8 vs 81.6 published; instruction language has no effect |

Also: 38 tests, and `speedrun.sh` verified end to end.

Verified infrastructure:

| Component | Status |
|---|---|
| Fresh clone runs the server | ✅ gRPC stubs regenerate when missing or version-mismatched |
| CI environment (`pixi run test`) | ✅ green on GitHub: 38 tests, 33 seconds |
| Docker image | ✅ 3.49 GB; the build tolerates an unreliable network |
| Docker with GPU | ✅ runs and was measured; 3–9% from native, below the noise threshold |

## Optional extensions

Not required; ordered by how worthwhile they are:

1. **Attention-score token pruning (FastV-style)** — none of the four current
   selection methods uses information from the model. Requires changes to the
   decoder loop.
2. **Combine edge 768 with nf4 and re-measure serving** — so far each has only been
   measured separately, offline.
3. **Add a second model** (e.g. Qwen2.5-VL-3B) to test whether "the vision encoder
   takes over half the time" holds beyond SmolVLM.
4. **Server-side batching** — the server currently handles one request at a time;
   batching could raise throughput on the same hardware.

## Notes for rerunning

- Never run two measurements on one GPU at the same time: latency jumped from 904 ms
  to 5,247 ms.
- Always pass `--model`; the default is now the 2.2B model, but be explicit anyway.
- `FAST=1 ./speedrun.sh` writes to `results/fast/` (ignored by git), so a pipeline
  check never overwrites reference results. The README figure is drawn from
  `results/gate2_sweep.json`: `pixi run python -m bench.plot --results results/gate2_sweep.json`
- Do not use `pkill -f` with a pattern that also matches the command being typed; it
  kills its own shell. Use a bracket pattern such as `"bench\.qu[a]ntize"`.
