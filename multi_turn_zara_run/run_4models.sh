#!/bin/bash
# Pipeline 1 (multi-turn) — 4-MODEL self-preference run.
#
# Generates all 4 models ONCE at the longest length (GEN_TURNS), then DERIVES the
# shorter turn counts by truncation (each turn only conditions on the history-so-far,
# so turns[:k] of an 8t convo is a valid kt convo -- see truncate_conversations.py).
# Then evaluates every EVAL_TURNS length for all C(4,2)=6 generator pairs, each judged
# by its OWN two models, and runs the self-preference (SPI) analysis per (pair, length).
# Fully unattended, no prompts, timestamped log.
#
# Usage:  bash multi_turn_zara_run/run_4models.sh [N] [GEN_TURNS] [T]
#           N         = scenarios              (default 125; powered for small effects)
#           GEN_TURNS = turns generated ONCE   (default 8; the longest length)
#           T         = max tokens/turn        (default 500)
#         EVAL_TURNS (the lengths actually evaluated) is the array below; edit it to
#         taste. Every entry must be <= GEN_TURNS. The entry equal to GEN_TURNS is the
#         freshly generated file; the rest are produced by truncation.
#         e.g. `bash multi_turn_zara_run/run_4models.sh 5 4`  for a quick smoke test
#              (generates 4t, evaluates whatever EVAL_TURNS entries are <= 4).
#
# Prereqs in .env:  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
#   Qwen (local): Ollama running with the model pulled; the OpenAI-compatible
#   endpoint defaults to http://localhost:11434/v1 (override via OPENAI_COMPAT_BASE_URL
#   / OPENAI_COMPAT_API_KEY to use OpenRouter/DashScope instead).
# On Sherlock: submit via the sbatch wrapper (run_4models.sbatch) so it runs on a
# compute node with a GPU for the local Ollama (Qwen). Do NOT run on the login node.

set -o pipefail
cd "$(dirname "$0")/.."
# Python interpreter: inherit from the environment (the sbatch wrapper activates a
# venv so `python` is on PATH). Override with `PY=/path/to/python bash run_4models.sh`.
PY=${PY:-python}

N=${1:-125}
GEN_TURNS=${2:-8}
T=${3:-500}

# Turn counts to evaluate. Each MUST be <= GEN_TURNS. The one equal to GEN_TURNS is
# the generated file; the strictly-smaller ones are derived by truncation.
EVAL_TURNS=(2 4 6 8)

# ---- Models (override via env; defaults are the exact API strings for this project) ----
# Native: gpt-*, claude-*, gemini-*.  OpenAI-compatible (Qwen/DeepSeek/local): anything else.
# The Qwen id MUST match the alias the sbatch wrapper serves via Ollama (':' -> '-'),
# which is why GEN_QWEN is exported by run_4models.sbatch (7b for smoke, 32b for full).
GEN_OPENAI=${GEN_OPENAI:-gpt-5.5}
GEN_ANTHROPIC=${GEN_ANTHROPIC:-claude-sonnet-5}
GEN_GEMINI=${GEN_GEMINI:-gemini-3.1-flash-lite}
GEN_QWEN=${GEN_QWEN:-qwen3.6-35b}
MODELS=("$GEN_ANTHROPIC" "$GEN_OPENAI" "$GEN_GEMINI" "$GEN_QWEN")
PATIENT=${PATIENT_MODEL:-$GEN_OPENAI}

GEN_DIR=multi_turn_zara_run/Generation
EVAL_DIR=multi_turn_zara_run/Evaluation
LOG="multi_turn_zara_run/run_4models_gen${GEN_TURNS}t_$(date +%Y%m%d_%H%M%S).log"

# Keep only EVAL_TURNS entries that fit within GEN_TURNS (drop any that exceed it).
FITTING_TURNS=()
for t in "${EVAL_TURNS[@]}"; do
  if [ "$t" -le "$GEN_TURNS" ]; then
    FITTING_TURNS+=("$t")
  else
    echo "  (note: eval length ${t}t > GEN_TURNS ${GEN_TURNS}t -- skipped)"
  fi
done

echo "=== 4-model run STARTED $(date) | N=$N GEN=${GEN_TURNS}t EVAL=${FITTING_TURNS[*]} T=$T | models: ${MODELS[*]} ===" | tee "$LOG"

# ---- Stage 1: generate ALL 4 models once, at the longest length ----
"$PY" src/generation/generate_conversations.py \
  --num_scenarios "$N" --turns "$GEN_TURNS" --shuffle --seed 42 \
  --models "${MODELS[@]}" \
  --patient_model "$PATIENT" \
  --max_tokens_per_turn "$T" \
  --output_dir "$GEN_DIR" >> "$LOG" 2>&1 \
&& echo "=== GENERATION done $(date) ===" | tee -a "$LOG" \
|| { echo "=== GENERATION FAILED $(date) ===" | tee -a "$LOG"; exit 1; }

# ---- Stage 1b: derive shorter lengths by truncation (targets strictly < GEN_TURNS) ----
TRUNC_TARGETS=()
for t in "${FITTING_TURNS[@]}"; do
  if [ "$t" -lt "$GEN_TURNS" ]; then
    TRUNC_TARGETS+=("$t")
  fi
done

if [ "${#TRUNC_TARGETS[@]}" -gt 0 ]; then
  INPUT_FILES=()
  for m in "${MODELS[@]}"; do
    INPUT_FILES+=("$GEN_DIR/${m}_${GEN_TURNS}t_conversations.json")
  done
  echo "=== TRUNCATION -> ${TRUNC_TARGETS[*]} $(date) ===" | tee -a "$LOG"
  "$PY" src/generation/truncate_conversations.py \
    --input_files "${INPUT_FILES[@]}" \
    --targets "${TRUNC_TARGETS[@]}" \
    --output_dir "$GEN_DIR" >> "$LOG" 2>&1 \
  && echo "=== TRUNCATION done $(date) ===" | tee -a "$LOG" \
  || { echo "=== TRUNCATION FAILED $(date) ===" | tee -a "$LOG"; exit 1; }
else
  echo "=== TRUNCATION skipped (only GEN_TURNS=${GEN_TURNS}t evaluated) $(date) ===" | tee -a "$LOG"
fi

# ---- Stage 2: evaluate every length for all 6 pairs (each judged by its own two models) + SPI ----
n=${#MODELS[@]}
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
    echo "=== PAIR: $A vs $B | judges: $A, $B | lengths: ${FITTING_TURNS[*]} | $(date) ===" | tee -a "$LOG"

    # One call covers all lengths x both judges (pairwise_evaluation.py grids --turns x --judge_models).
    "$PY" src/evaluation/pairwise_evaluation.py \
      --conv_dir "$GEN_DIR" \
      --gen_a "$A" --gen_b "$B" \
      --turns "${FITTING_TURNS[@]}" \
      --judge_models "$A" "$B" \
      --scenarios "$GEN_DIR/scenarios.json" \
      --output_dir "$EVAL_DIR" >> "$LOG" 2>&1

    # Self-preference analysis per length (judge files are named by judge id + turn count).
    for TURNS in "${FITTING_TURNS[@]}"; do
      JA="$EVAL_DIR/pairwise_${A}_vs_${B}_${TURNS}t_${A}_Judge.json"
      JB="$EVAL_DIR/pairwise_${A}_vs_${B}_${TURNS}t_${B}_Judge.json"
      if [ -f "$JA" ] && [ -f "$JB" ]; then
        "$PY" src/evaluation/analyze_self_preference_pairwise.py \
          --judge_files "$JA" "$JB" \
          --output "$EVAL_DIR/self_preference_${A}_vs_${B}_${TURNS}t.json" >> "$LOG" 2>&1
      else
        echo "  (${TURNS}t: skipped SPI: one/both judge files missing)" | tee -a "$LOG"
      fi
    done
  done
done

echo "=== ALL DONE $(date). Log: $LOG ===" | tee -a "$LOG"
