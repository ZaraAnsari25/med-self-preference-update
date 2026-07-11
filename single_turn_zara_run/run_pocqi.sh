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
#                  THINK=0|1            (0=no-think [default], 1=think). Selects BOTH the
#                                        output directory AND the models' thinking mode, so
#                                        the two studies' data stay separate:
#                                          THINK=0 -> single_turn_zara_run/nothink/...
#                                          THINK=1 -> single_turn_zara_run/think/...
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

# ---- Thinking mode -> output dir + per-model thinking knobs ----
# Knobs are read by the model clients (generation + judges); defaults keep no-think, so
# existing pipelines are unaffected. THINK=1 turns thinking on consistently everywhere.
THINK=${THINK:-0}
if [ "$THINK" = "1" ]; then
  MODE=think
  export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-medium}"   # gpt-5.x/o-series
  export GEMINI_THINKING_BUDGET="${GEMINI_THINKING_BUDGET:--1}"          # gemini: -1 = dynamic
  export QWEN_THINK="${QWEN_THINK:-1}"                                   # qwen: enable thinking
  export CLAUDE_THINKING_EFFORT="${CLAUDE_THINKING_EFFORT:-medium}"      # claude 5: adaptive @ effort
else
  MODE=nothink
  export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-none}"
  export GEMINI_THINKING_BUDGET="${GEMINI_THINKING_BUDGET:-0}"
  export QWEN_THINK="${QWEN_THINK:-0}"
  export CLAUDE_THINKING_EFFORT="${CLAUDE_THINKING_EFFORT:-none}"        # claude 5: thinking OFF
fi

RUN_DIR=single_turn_zara_run/$MODE
GEN_DIR=$RUN_DIR/Generation
EVAL_DIR=$RUN_DIR/Evaluation
LOG="$RUN_DIR/run_pocqi_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$GEN_DIR" "$EVAL_DIR"

echo "=== POCQi single-turn run STARTED $(date) | MODE=$MODE (THINK=$THINK) N=$N MAXTOK=$MAXTOK specialty='${SPECIALTY:-ALL}' | models: ${MODELS[*]} ===" | tee "$LOG"
echo "    thinking knobs: OPENAI_REASONING_EFFORT=$OPENAI_REASONING_EFFORT GEMINI_THINKING_BUDGET=$GEMINI_THINKING_BUDGET QWEN_THINK=$QWEN_THINK CLAUDE_THINKING_EFFORT=$CLAUDE_THINKING_EFFORT | out=$RUN_DIR" | tee -a "$LOG"

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
# Parallelized by JUDGE: each judge (a distinct provider) runs in its own background
# subshell and scores its generators SEQUENTIALLY, so any one provider still sees at most
# MAX_CONCURRENCY calls at once (unchanged) while the 4 providers overlap -> ~4x. Errors
# are best-effort (logged), matching the prior sequential behavior.
for judge in "${MODELS[@]}"; do
  (
    for gen in "${MODELS[@]}"; do
      [ -f "$(rf "$gen")" ] || continue
      "$PY" src/evaluation/evaluate_pocqi.py --mode individual \
        --response_file "$(rf "$gen")" --judge_model "$judge" \
        --max_concurrency "$MAX_CONCURRENCY" \
        --output "$EVAL_DIR/individual_${gen}_by_${judge}.json" >> "$LOG" 2>&1
    done
  ) &
done
wait
echo "=== INDIVIDUAL scoring done $(date) ===" | tee -a "$LOG"

# ---- Stage 3: PAIRWISE for all C(4,2)=6 pairs, each judged by its own two models, + SPI ----
n=${#MODELS[@]}

# Phase 3a: run every pairwise JUDGE evaluation, grouped by JUDGE. Each judge (distinct
# provider) runs its pairs SEQUENTIALLY in a background subshell, so per-provider load
# stays at MAX_CONCURRENCY (unchanged) while the 4 judges overlap -> ~4x. A judge only
# scores the pairs it belongs to (every pair is judged by BOTH its models). The A/B swap
# is a pure function of (--seed, scenario_id), so it is unaffected by this parallelism.
for judge in "${MODELS[@]}"; do
  (
    for ((i=0; i<n; i++)); do
      for ((j=i+1; j<n; j++)); do
        A=${MODELS[$i]}; B=${MODELS[$j]}
        [ "$judge" = "$A" ] || [ "$judge" = "$B" ] || continue
        [ -f "$(rf "$A")" ] && [ -f "$(rf "$B")" ] || continue
        "$PY" src/evaluation/evaluate_pocqi.py --mode pairwise \
          --response_file_a "$(rf "$A")" --response_file_b "$(rf "$B")" \
          --judge_model "$judge" --seed 42 --max_concurrency "$MAX_CONCURRENCY" \
          --output "$EVAL_DIR/pairwise_${A}_vs_${B}_${judge}_Judge.json" >> "$LOG" 2>&1
      done
    done
  ) &
done
wait
echo "=== PAIRWISE judging done $(date) ===" | tee -a "$LOG"

# Phase 3b: per-pair self-preference analysis (CPU-only, fast). Needs BOTH judge files.
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
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
    else
      echo "  (skip analysis $A vs $B: missing one/both judge files)" | tee -a "$LOG"
    fi
  done
done
echo "=== PAIRWISE + SPI done $(date). Log: $LOG ===" | tee -a "$LOG"
