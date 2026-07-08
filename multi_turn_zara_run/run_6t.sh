#!/bin/bash
# Pipeline 1 (multi-turn) — 6t generation + evaluation, fully unattended.
#
# Usage:  bash multi_turn_zara_run/run_6t.sh [N] [T]
#           N = number of scenarios (default 200)
#           T = max tokens per turn  (default 500)
#         e.g. `bash multi_turn_zara_run/run_6t.sh 10`  for a quick 10-scenario test.
# Keep the laptop LID OPEN, plugged into power, on stable WiFi.
# `caffeinate -i` prevents idle sleep while it runs (it does NOT override lid-close sleep).
#
# Runs to completion with NO interactive prompts. All output is timestamped to a log file.

set -o pipefail
cd "$(dirname "$0")/.."          # move to project root regardless of where it's run from

PY=/opt/anaconda3/bin/python     # the interpreter that has the deps installed
N=${1:-200}                      # scenarios (both models, same set)
T=${2:-500}                      # max_completion_tokens per turn
LOG="multi_turn_zara_run/run_6t_$(date +%Y%m%d_%H%M%S).log"

echo "=== 6t run STARTED $(date) | N=$N T=$T ===" | tee "$LOG"

# --- Stage 1: generation (GPT-5.5 + Claude Sonnet 5) ---
caffeinate -i "$PY" src/generation/generate_conversations.py \
  --num_scenarios "$N" --turns 6 --shuffle --seed 42 \
  --models gpt-5.5 claude-sonnet-5 \
  --patient_model gpt-5.5 \
  --max_tokens_per_turn "$T" \
  --output_dir multi_turn_zara_run/Generation >> "$LOG" 2>&1 \
&& echo "=== GENERATION done $(date); starting EVALUATION ===" | tee -a "$LOG" \
&& caffeinate -i "$PY" src/evaluation/pairwise_evaluation.py \
  --conv_dir multi_turn_zara_run/Generation \
  --gen_a gpt-5.5 --gen_b claude-sonnet-5 \
  --turns 6 \
  --judge_models claude-sonnet-5 gpt-5.5 \
  --scenarios multi_turn_zara_run/Generation/scenarios.json \
  --output_dir multi_turn_zara_run/Evaluation >> "$LOG" 2>&1 \
&& echo "=== ALL DONE $(date) ===" | tee -a "$LOG"

echo "=== script exited (code $?) at $(date). Log: $LOG ===" | tee -a "$LOG"
