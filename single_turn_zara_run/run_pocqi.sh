#!/bin/bash
# Pipeline 2 (single-turn) — Real-POCQi generalist->specialist self-preference run.
#
# Generates one SPECIALIST answer per question per model, then evaluates BOTH ways
# (like the COVID run): pairwise head-to-head + individual absolute scoring, and computes
# the Self-Preference Index for each. Additive: does not touch the multi-turn pipeline.
#
# Usage:  bash single_turn_zara_run/run_pocqi.sh [N] [MAXTOK]
#           N       = number of questions   (default 125; all=620)
#           MAXTOK  = max tokens per answer  (default 1024)
#   Optional env:  SPECIALTY="Cardiology,Neurology"  (restrict specialties; default all)
#                  MAX_CONCURRENCY=12   (cloud fan-out; Qwen gated by OLLAMA_NUM_PARALLEL)
#
# On Sherlock submit via run_pocqi.sbatch (starts local Ollama for Qwen).

set -o pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

N=${1:-125}
MAXTOK=${2:-1024}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-12}
SPECIALTY_ARG=()
[ -n "${SPECIALTY:-}" ] && SPECIALTY_ARG=(--specialty "$SPECIALTY")

# Specialist models under study (each answers every question). Same 4 as multi-turn.
MODELS=(claude-sonnet-5 gpt-5.5 gemini-3.1-flash-lite qwen3.6-35b)

GEN_DIR=single_turn_zara_run/Generation
EVAL_DIR=single_turn_zara_run/Evaluation
LOG="single_turn_zara_run/run_pocqi_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$GEN_DIR" "$EVAL_DIR"

echo "=== POCQi single-turn run STARTED $(date) | N=$N MAXTOK=$MAXTOK specialty='${SPECIALTY:-ALL}' | models: ${MODELS[*]} ===" | tee "$LOG"

# ---- Stage 1: generate specialist answers (all models) ----
"$PY" src/generation/generate_single_turn_pocqi.py \
  --models "${MODELS[@]}" \
  --num_questions "$N" --shuffle --seed 42 \
  --max_tokens "$MAXTOK" --max_concurrency "$MAX_CONCURRENCY" \
  "${SPECIALTY_ARG[@]}" \
  --output_dir "$GEN_DIR" >> "$LOG" 2>&1 \
&& echo "=== GENERATION done $(date) ===" | tee -a "$LOG" \
|| { echo "=== GENERATION FAILED $(date) ===" | tee -a "$LOG"; exit 1; }

rf() { echo "$GEN_DIR/$1_pocqi_responses.json"; }

# ---- Stage 2: INDIVIDUAL absolute scoring — every generator by every judge ----
for gen in "${MODELS[@]}"; do
  for judge in "${MODELS[@]}"; do
    [ -f "$(rf "$gen")" ] || continue
    "$PY" src/evaluation/evaluate_pocqi.py --mode individual \
      --response_file "$(rf "$gen")" --judge_model "$judge" \
      --max_concurrency "$MAX_CONCURRENCY" \
      --output "$EVAL_DIR/individual_${gen}_by_${judge}.json" >> "$LOG" 2>&1
  done
done
echo "=== INDIVIDUAL scoring done $(date) ===" | tee -a "$LOG"

# ---- Stage 3: PAIRWISE for all C(4,2)=6 pairs, each judged by its own two models, + SPI ----
n=${#MODELS[@]}
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
    [ -f "$(rf "$A")" ] && [ -f "$(rf "$B")" ] || { echo "  (skip pair $A vs $B: missing responses)" | tee -a "$LOG"; continue; }
    echo "=== PAIR: $A vs $B | judges: $A, $B | $(date) ===" | tee -a "$LOG"
    for judge in "$A" "$B"; do
      "$PY" src/evaluation/evaluate_pocqi.py --mode pairwise \
        --response_file_a "$(rf "$A")" --response_file_b "$(rf "$B")" \
        --judge_model "$judge" --seed 42 --max_concurrency "$MAX_CONCURRENCY" \
        --output "$EVAL_DIR/pairwise_${A}_vs_${B}_${judge}_Judge.json" >> "$LOG" 2>&1
    done
    # Pairwise SPI for this pair (reuse the multi-turn analyzer, unchanged).
    JA="$EVAL_DIR/pairwise_${A}_vs_${B}_${A}_Judge.json"
    JB="$EVAL_DIR/pairwise_${A}_vs_${B}_${B}_Judge.json"
    if [ -f "$JA" ] && [ -f "$JB" ]; then
      "$PY" src/evaluation/analyze_self_preference_pairwise.py \
        --judge_files "$JA" "$JB" \
        --output "$EVAL_DIR/self_preference_pairwise_${A}_vs_${B}.json" >> "$LOG" 2>&1
      # Per-model self-preference bias with 95% CI + p (headline metric).
      "$PY" src/evaluation/spi_significance.py \
        --judge_files "$JA" "$JB" \
        --output "$EVAL_DIR/spi_significance_${A}_vs_${B}.json" >> "$LOG" 2>&1
      # Decision-level self-preference: per-judge own/other/tie + binomial test.
      "$PY" src/evaluation/preference_breakdown.py \
        --judge_files "$JA" "$JB" \
        --output "$EVAL_DIR/preference_${A}_vs_${B}.json" >> "$LOG" 2>&1
    fi
  done
done
echo "=== PAIRWISE + SPI done $(date). Log: $LOG ===" | tee -a "$LOG"
