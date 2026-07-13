#!/bin/bash
# Slurm wrapper for Pipeline 2 (single-turn, Real-POCQi generalist->specialist) on Sherlock.
# Starts a local Ollama (for the Qwen specialist/judge), then runs run_pocqi.sh
# (generate -> individual + pairwise eval -> SPI). All localhost on one GPU node.
#
# SUBMIT (from project root):
#   sbatch single_turn_zara_run/run_pocqi.sbatch            # full run, no-think (-> nothink/)
#   THINK=1 sbatch single_turn_zara_run/run_pocqi.sbatch    # full run, think mode (-> think/)
#   SMOKE=1 sbatch single_turn_zara_run/run_pocqi.sbatch    # smoke: 4 questions, all 4 models
#   THINK/SMOKE inherit via Slurm's default --export=ALL; combine them freely.
# Watch:  squeue --me     Logs:  single_turn_zara_run/slurm_<jobid>.out
#
#SBATCH -p roxanad
#SBATCH -G 1
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH -t 08:00:00
#SBATCH -J pocqi-single
#SBATCH -o single_turn_zara_run/slurm_%j.out
#SBATCH -e single_turn_zara_run/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zansari6@stanford.edu

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "=== SBATCH start $(date) on $SLURMD_NODENAME (job $SLURM_JOB_ID) ==="

QWEN_PULL="qwen3.6:35b"
QWEN_ALIAS="${QWEN_PULL/:/-}"   # qwen3.6-35b (matches run_pocqi.sh MODELS)

export HF_HOME=$SCRATCH/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCH/.cache
export OLLAMA_MODELS=$SCRATCH/ollama/models
mkdir -p "$HF_HOME" "$OLLAMA_MODELS"

# Concurrency (cloud fan-out; Qwen served concurrently on the H100).
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-12}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-12}"

module load python/3.12.1 ollama/0.18.0
source "$SCRATCH/venvs/med-self-pref/bin/activate"
export PY=python
set -a; source .env; set +a

# Job-specific Ollama port so two Ollama jobs on the SAME node never collide on 11434.
# Set AFTER sourcing .env so it wins, and point the OpenAI-compat clients at it (this
# sbatch always serves Ollama locally, so the compat base URL must be this local port).
OLLAMA_PORT=$(( 11434 + SLURM_JOB_ID % 20000 ))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OPENAI_COMPAT_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1"
export OPENAI_COMPAT_API_KEY="ollama"
echo "Ollama on ${OLLAMA_HOST} (job-specific port)"

ollama serve > "single_turn_zara_run/ollama_${SLURM_JOB_ID}.log" 2>&1 &
trap 'kill %1 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 || { echo "ERROR: Ollama did not start"; exit 1; }
ollama pull "$QWEN_PULL"        || { echo "ERROR: ollama pull failed"; exit 1; }
ollama cp "$QWEN_PULL" "$QWEN_ALIAS" 2>/dev/null || true
echo "Qwen served as alias: $QWEN_ALIAS"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "=== SMOKE: 4 questions, all 4 models ==="
  bash single_turn_zara_run/run_pocqi.sh 4 800
else
  bash single_turn_zara_run/run_pocqi.sh    # full run: 125 questions, 1024 tokens
fi
echo "=== SBATCH done $(date) ==="
