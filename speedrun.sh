#!/usr/bin/env bash
# One command reruns every result in the project, from tests to figures.
#
#   ./speedrun.sh          # full run, about 45–60 minutes on a 6 GB GPU
#   FAST=1 ./speedrun.sh   # reduced run to check the pipeline, about 6 minutes
set -euo pipefail

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
MODEL="${MODEL:-HuggingFaceTB/SmolVLM-Instruct}"
if [[ "${FAST:-0}" == "1" ]]; then
  # A pipeline check writes to its own folder so it never overwrites reference results.
  PROBE_RUNS=15; SAMPLES=12; ROUNDS=1; BREAKDOWN_SAMPLES=4; OUT=results/fast
else
  PROBE_RUNS=60; SAMPLES=100; ROUNDS=3; BREAKDOWN_SAMPLES=12; OUT=results
fi
mkdir -p "$OUT"

echo "== 0/5 tests =="
$PIXI run python -m pytest tests/ -q

echo "== 1/5 gate 0: noise of the measurement itself =="
$PIXI run python bench/latency_probe.py --model "$MODEL" --runs "$PROBE_RUNS" \
  --out "$OUT/gate0_latency.json"
# The improvement threshold uses the noise just measured, on the same model (rule 5).
NOISE_CV=$($PIXI run python -c "import json; print(json.load(open('$OUT/gate0_latency.json'))['generate_ms']['cv_pct'])")

echo "== 2/5 latency breakdown by component =="
$PIXI run python -m bench.breakdown --model "$MODEL" \
  --samples "$BREAKDOWN_SAMPLES" --out "$OUT/breakdown.json"

echo "== 3/5 lever sweep: fewer image tiles versus token pruning =="
$PIXI run python -m bench.harness --model "$MODEL" \
  --samples "$SAMPLES" --rounds "$ROUNDS" --noise-cv "$NOISE_CV" \
  --configs "baseline,edge1152,edge960,edge768,edge576,nosplit,keep0.5:uniform,keep0.25:uniform,keep0.25:random,keep0.077:uniform" \
  --out "$OUT/sweep.json"

echo "== 4/5 trade-off figure =="
$PIXI run python -m bench.plot --results "$OUT/sweep.json" --stem "$OUT/tradeoff"

echo "== 5/5 done. Results are in $OUT/ =="
