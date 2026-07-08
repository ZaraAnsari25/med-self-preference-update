#!/bin/bash
# Pipeline 1 (multi-turn) — 4-MODEL self-preference run.
#
# Generates all 4 models ONCE, then evaluates all C(4,2)=6 generator pairs,
# each judged by its OWN two models, then runs the self-preference (SPI) analysis
# per pair. Fully unattended, no prompts, timestamped log.
#
# Usage:  bash multi_turn_zara_run/run_4models.sh [N] [TURNS] [T]
#           N     = scenarios       (default 200)
#           TURNS = turns/conversation (default 2)
#           T     = max tokens/turn  (default 500)
#         e.g. `bash multi_turn_zara_run/run_4models.sh 5 2`  for a quick smoke test.
#
# Prereqs in .env:  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
#   Qwen (local): Ollama running with the model pulled; the OpenAI-compatible
#   endpoint defaults to http://localhost:11434/v1 (override via OPENAI_COMPAT_BASE_URL
#   / OPENAI_COMPAT_API_KEY to use OpenRouter/DashScope instead).
# Keep the laptop LID OPEN, plugged in, on WiFi. caffeinate blocks idle sleep.

set -o pipefail
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/bin/python

N=${1:-200}
TURNS=${2:-2}
T=${3:-500}

# ---- Models (EDIT these ids to match your setup / exact API strings) ----
# Native: gpt-*, claude-*, gemini-*.  OpenAI-compatible (Qwen/DeepSeek/local): anything else.
# NOTE: avoid ':' in the Qwen id (Ollama's default tag is qwen3.6:35b) -- ':' makes ugly
# filenames. Alias it once:  `ollama cp qwen3.6:35b qwen3.6-35b`  and use the alias below.
MODELS=(claude-sonnet-5 gpt-5.5 gemini-3.5 qwen3.6-35b)
PATIENT=gpt-5.5

GEN_DIR=multi_turn_zara_run/Generation
EVAL_DIR=multi_turn_zara_run/Evaluation
LOG="multi_turn_zara_run/run_4models_${TURNS}t_$(date +%Y%m%d_%H%M%S).log"

echo "=== 4-model ${TURNS}t run STARTED $(date) | N=$N T=$T | models: ${MODELS[*]} ===" | tee "$LOG"

# ---- Stage 1: generate ALL 4 models once (same scenarios) ----
caffeinate -i "$PY" src/generation/generate_conversations.py \
  --num_scenarios "$N" --turns "$TURNS" --shuffle --seed 42 \
  --models "${MODELS[@]}" \
  --patient_model "$PATIENT" \
  --max_tokens_per_turn "$T" \
  --output_dir "$GEN_DIR" >> "$LOG" 2>&1 \
&& echo "=== GENERATION done $(date) ===" | tee -a "$LOG" \
|| { echo "=== GENERATION FAILED $(date) ===" | tee -a "$LOG"; exit 1; }

# ---- Stage 2: evaluate all 6 pairs (each judged by its own two models) + SPI ----
n=${#MODELS[@]}
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
    echo "=== PAIR: $A vs $B | judges: $A, $B | $(date) ===" | tee -a "$LOG"

    caffeinate -i "$PY" src/evaluation/pairwise_evaluation.py \
      --conv_dir "$GEN_DIR" \
      --gen_a "$A" --gen_b "$B" \
      --turns "$TURNS" \
      --judge_models "$A" "$B" \
      --scenarios "$GEN_DIR/scenarios.json" \
      --output_dir "$EVAL_DIR" >> "$LOG" 2>&1

    # Self-preference analysis for this pair (judge files are named by judge id).
    JA="$EVAL_DIR/pairwise_${A}_vs_${B}_${TURNS}t_${A}_Judge.json"
    JB="$EVAL_DIR/pairwise_${A}_vs_${B}_${TURNS}t_${B}_Judge.json"
    if [ -f "$JA" ] && [ -f "$JB" ]; then
      "$PY" src/evaluation/analyze_self_preference_pairwise.py \
        --judge_files "$JA" "$JB" \
        --output "$EVAL_DIR/self_preference_${A}_vs_${B}_${TURNS}t.json" >> "$LOG" 2>&1
    else
      echo "  (skipped SPI: one/both judge files missing)" | tee -a "$LOG"
    fi
  done
done

echo "=== ALL DONE $(date). Log: $LOG ===" | tee -a "$LOG"
