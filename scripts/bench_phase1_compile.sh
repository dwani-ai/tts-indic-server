#!/usr/bin/env bash
# Phase 1 benchmark: bfloat16 + warmup vs float32 baseline.
# torch.compile is disabled by default (incompatible with IndicF5/torchdiffeq).
# Usage: from repo root, with venv active and prompts/ available:
#   bash scripts/bench_phase1_compile.sh

set -e
cd "$(dirname "$0")/.."
SERVER_DIR="${1:-src/server}"
cd "$SERVER_DIR"

echo "=============================================="
echo "Phase 1 benchmark: bfloat16 vs float32"
echo "=============================================="
echo ""

echo ">>> Phase 1 (bfloat16 + warmup)"
python3 tts_indic_f5.py --warmup-runs 2 --timed-runs 3
echo ""

echo ">>> Baseline (float32, same warmup)"
python3 tts_indic_f5.py --float32 --warmup-runs 2 --timed-runs 3
echo ""

echo "=============================================="
echo "Compare the 'time (avg)' and 'RTF' lines above."
echo "=============================================="
