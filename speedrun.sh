#!/usr/bin/env bash
# One command reruns every result in the README, then regenerates its tables.
#
#   ./speedrun.sh            # full run, about 5 hours on a 6 GB GPU
#   FAST=1 ./speedrun.sh     # reduced run to check the pipeline, about 15 minutes
#   DOCKER=1 ./speedrun.sh   # also build the image and measure the server in Docker
#   START=5 ./speedrun.sh    # resume from step 5 after a failure (END=n stops after step n)
#
# The GPU must be idle: every timing script refuses to start otherwise, and the
# harness fails a run in which another process took GPU memory.
set -euo pipefail

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
MODEL="${MODEL:-HuggingFaceTB/SmolVLM-Instruct}"
PY="$PIXI run python"
# The server runs as a direct child, so `kill` reaches the process that holds the GPU.
PYBIN=$($PIXI run python -c "import sys; print(sys.executable)")

if [[ "${FAST:-0}" == "1" ]]; then
  # A pipeline check writes to its own folder so it never overwrites reference results.
  OUT=results/fast
  PROBE_RUNS=15; BREAKDOWN_SAMPLES=3; PREP_SAMPLES=4
  CONFIRM_SAMPLES=12; NF4_SAMPLES=12; SWEEP_SAMPLES=8; PROMPT_SAMPLES=8; DOCVQA_SAMPLES=8
  QUANT_SAMPLES=4; SERVE_REQUESTS=8; ROUNDS=2
else
  OUT=results
  PROBE_RUNS=60; BREAKDOWN_SAMPLES=12; PREP_SAMPLES=16
  # step 4 uses the whole ChartQA validation split, the most questions available.
  # Power of the exact McNemar test (alpha = 0.05) for edge 768's measured loss, with
  # 13.4% of questions discordant and 58.1% of those favouring the baseline:
  # 0.14 at 300 questions, 0.73 at 1,920; 80% would need about 2,300. The README
  # section "Why the whole split" computes these from the results (bench.metrics).
  CONFIRM_SAMPLES=1920; NF4_SAMPLES=300; SWEEP_SAMPLES=100; PROMPT_SAMPLES=200; DOCVQA_SAMPLES=300
  QUANT_SAMPLES=16; SERVE_REQUESTS=40; ROUNDS=2
fi
mkdir -p "$OUT"
START="${START:-0}"          # resume: START=5 ./speedrun.sh skips steps 0-4
END="${END:-9}"              # a range: START=4 END=4 ./speedrun.sh reruns step 4 only
step() { [[ "$1" -ge "$START" && "$1" -le "$END" ]]; }
LEVERS="baseline,edge1152,edge960,edge768,edge576,nosplit"
LEVERS="$LEVERS,keep0.5:uniform,keep0.25:uniform,keep0.25:random,keep0.25:pool,keep0.25:norm,keep0.077:uniform"

if step 0; then
  echo "== 0/9 tests =="
  $PY -m pytest tests/ -q
fi

if step 1; then
  echo "== 1/9 gate 0: noise of the measurement itself =="
  $PY -m bench.latency_probe --model "$MODEL" --runs "$PROBE_RUNS" --out "$OUT/gate0_latency.json"
fi
# The noise measured in step 1, on the same model, is recorded with every harness run.
NOISE_CV=$($PY -c "import json; from bench.metrics import robust_cv; print(robust_cv(json.load(open('$OUT/gate0_latency.json'))['raw_generate_ms']))")

if step 2; then
  echo "== 2/9 where the time goes =="
  $PY -m bench.breakdown --model "$MODEL" --samples "$BREAKDOWN_SAMPLES" --out "$OUT/breakdown.json"
  $PY -m bench.preprocess_cost --model "$MODEL" --samples "$PREP_SAMPLES" --out "$OUT/preprocess_cost.json"
fi

if step 3; then
  echo "== 3/9 lever sweep: fewer tiles versus token pruning =="
  $PY -m bench.harness --model "$MODEL" --samples "$SWEEP_SAMPLES" --rounds "$ROUNDS" \
    --noise-cv "$NOISE_CV" --configs "$LEVERS" --out "$OUT/gate2_sweep.json"
fi

if step 4; then
  echo "== 4/9 the main lever on the whole validation split =="
  $PY -m bench.harness --model "$MODEL" --samples "$CONFIRM_SAMPLES" --rounds "$ROUNDS" \
    --noise-cv "$NOISE_CV" --configs "baseline,edge768" --out "$OUT/gate2_confirm.json"
fi

if step 5; then
  echo "== 5/9 quantisation: latency, memory, logit parity =="
  $PY -m bench.quantize --model "$MODEL" --samples "$QUANT_SAMPLES" --out "$OUT/gate3_quant.json"
fi

if step 6; then
  echo "== 6/9 nf4 accuracy on the first questions of step 4 =="
  $PY -m bench.harness --model "$MODEL" --quant nf4 --samples "$NF4_SAMPLES" --rounds "$ROUNDS" \
    --noise-cv "$NOISE_CV" --configs "baseline,edge768" --out "$OUT/gate5_nf4.json"
fi

if step 7; then
  echo "== 7/9 instruction language, and the DocVQA check against the published score =="
  $PY -m bench.harness --model "$MODEL" --samples "$PROMPT_SAMPLES" --rounds "$ROUNDS" \
    --noise-cv "$NOISE_CV" --configs "baseline,prompten" --out "$OUT/gate6_prompt.json"
  $PY -m bench.sanity_docvqa --model "$MODEL" --samples "$DOCVQA_SAMPLES" --prompt vi \
    --out "$OUT/sanity_docvqa.json"
  $PY -m bench.sanity_docvqa --model "$MODEL" --samples "$DOCVQA_SAMPLES" --prompt en \
    --out "$OUT/sanity_docvqa_en.json"
fi

serve_bench() {   # $1 = longest edge, $2 = output file
  local edge="$1" out="$2"
  "$PYBIN" -m serve.server --model "$MODEL" --max-edge "$edge" > "$OUT/server_$edge.log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT      # expands now: stop this server on any exit
  $PY -m serve.client_bench --samples "$SERVE_REQUESTS" --concurrency 1,2,4 \
    --max-edge "$edge" --out "$out"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  trap - EXIT
}
if step 8; then
  echo "== 8/9 serving over gRPC =="
  $PY -c "from bench.latency_probe import check_gpu_idle as c; m = c(); exit(m) if m else None"
  serve_bench 1536 "$OUT/gate4_serving_1536.json"
  serve_bench 768 "$OUT/gate4_serving_768.json"
  if [[ "${DOCKER:-0}" == "1" ]]; then
    echo "== 8b/9 the same server inside Docker =="
    docker build -t vlm-speedrun .
    for edge in 1536 768; do
      cid=$(docker run -d --gpus all -p 50051:50051 -v ~/.cache/huggingface:/models \
            vlm-speedrun pixi run python -m serve.server --max-edge "$edge")
      $PY -m serve.client_bench --samples "$SERVE_REQUESTS" --concurrency 1,2,4 \
        --max-edge "$edge" --out "$OUT/gate4_docker_$edge.json"
      docker rm -f "$cid" > /dev/null
    done
  fi
fi

echo "== 9/9 figure and README tables =="
$PY -m bench.plot --results "$OUT/gate2_sweep.json" --stem "$OUT/tradeoff"
if [[ "$OUT" == "results" ]]; then
  $PY -m bench.report --write
fi
echo "== done. Results are in $OUT/ =="
