#!/usr/bin/env bash
# One command reruns every result in the project, from tests to figures.
#
#   ./speedrun.sh          # full run, about 45–60 minutes on a 6 GB GPU
#   FAST=1 ./speedrun.sh   # reduced run to check the pipeline, about 6 minutes
set -euo pipefail

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
MODEL="${MODEL:-HuggingFaceTB/SmolVLM-Instruct}"
if [[ "${FAST:-0}" == "1" ]]; then
  PROBE_RUNS=15; SAMPLES=12; ROUNDS=1; BREAKDOWN_SAMPLES=4
else
  PROBE_RUNS=60; SAMPLES=100; ROUNDS=3; BREAKDOWN_SAMPLES=12
fi

echo "== 0/5 tests =="
$PIXI run python -m pytest tests/ -q

echo "== 1/5 gate 0: noise of the measurement itself =="
$PIXI run python bench/latency_probe.py --runs "$PROBE_RUNS" \
  --out results/gate0_latency.json

echo "== 2/5 latency breakdown by component =="
$PIXI run python -m bench.breakdown --model "$MODEL" \
  --samples "$BREAKDOWN_SAMPLES" --out results/breakdown.json

echo "== 3/5 lever sweep: fewer image tiles versus token pruning =="
$PIXI run python -m bench.harness --model "$MODEL" \
  --samples "$SAMPLES" --rounds "$ROUNDS" \
  --configs "baseline,edge1152,edge960,edge768,edge576,nosplit,keep0.5:uniform,keep0.25:uniform,keep0.25:random,keep0.077:uniform" \
  --out results/sweep.json

echo "== 4/5 trade-off figure =="
$PIXI run python -m bench.plot --results results/sweep.json --stem results/tradeoff

echo "== 5/5 done. Results are in results/ =="
