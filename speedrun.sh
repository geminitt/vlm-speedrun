#!/usr/bin/env bash
# Một lệnh chạy lại toàn bộ kết quả của dự án, từ kiểm thử tới biểu đồ.
#
#   ./speedrun.sh          # bản đầy đủ, khoảng 45–60 phút trên GPU 6 GB
#   FAST=1 ./speedrun.sh   # bản rút gọn để kiểm tra pipeline, khoảng 6 phút
set -euo pipefail

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
MODEL="${MODEL:-HuggingFaceTB/SmolVLM-Instruct}"
if [[ "${FAST:-0}" == "1" ]]; then
  PROBE_RUNS=15; SAMPLES=12; ROUNDS=1; BREAKDOWN_SAMPLES=4
else
  PROBE_RUNS=60; SAMPLES=100; ROUNDS=3; BREAKDOWN_SAMPLES=12
fi

echo "== 0/5 kiểm thử =="
$PIXI run python -m pytest tests/ -q

echo "== 1/5 cổng 0: độ nhiễu của phép đo =="
$PIXI run python bench/latency_probe.py --runs "$PROBE_RUNS" \
  --out results/gate0_latency.json

echo "== 2/5 bóc tách thời gian theo thành phần =="
$PIXI run python -m bench.breakdown --model "$MODEL" \
  --samples "$BREAKDOWN_SAMPLES" --out results/breakdown.json

echo "== 3/5 quét đòn bẩy: giảm số ô ảnh so với cắt token =="
$PIXI run python -m bench.harness --model "$MODEL" \
  --samples "$SAMPLES" --rounds "$ROUNDS" \
  --configs "baseline,edge1152,edge960,edge768,edge576,nosplit,keep0.5:uniform,keep0.25:uniform,keep0.25:random,keep0.077:uniform" \
  --out results/sweep.json

echo "== 4/5 vẽ biểu đồ đánh đổi =="
$PIXI run python -m bench.plot --results results/sweep.json --stem results/tradeoff

echo "== 5/5 xong. Kết quả nằm trong results/ =="
