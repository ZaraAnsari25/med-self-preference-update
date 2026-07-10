#!/bin/bash
# Slurm wrapper for Pipeline 1 (multi-turn) 4-model run on Sherlock.
#
# Runs the WHOLE pipeline inside ONE gpu-partition job:
#   1. starts a local Ollama server (serves the Qwen model on localhost:11434)
#   2. pulls the Qwen model to $SCRATCH
#   3. activates the venv + loads API keys from .env
#   4. runs multi_turn_zara_run/run_4models.sh (generate -> truncate -> eval -> SPI)
# Everything is localhost, so no cross-node networking.
#
# SUBMIT (from project root):
#   sbatch multi_turn_zara_run/run_4models.sbatch            # full run (defaults in run_4models.sh)
#   SMOKE=1 sbatch multi_turn_zara_run/run_4models.sbatch    # smoke test (2 scenarios, real 8t design)
# Watch:  squeue --me      Logs:  multi_turn_zara_run/slurm_<jobid>.out
#
#SBATCH -p roxanad
#SBATCH -G 1
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH -t 04:00:00
#SBATCH -J med-selfpref
#SBATCH -o multi_turn_zara_run/slurm_%j.out
#SBATCH -e multi_turn_zara_run/slurm_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "=== SBATCH start $(date) on $SLURMD_NODENAME (job $SLURM_JOB_ID) ==="

# ---- Qwen model served by Ollama. Must match the qwen id in run_4models.sh MODELS. ----
# qwen3.6:35b (23.9 GB) verified pullable from the Ollama registry; cached in $SCRATCH
# after first pull, so smoke and full runs reuse it. Needs the 40GB GPU requested above.
QWEN_PULL="qwen3.6:35b"
QWEN_ALIAS="${QWEN_PULL/:/-}"   # ':' -> '-' -> qwen3.6-35b (matches run_4models.sh GEN_QWEN)

# ---- Caches off $HOME (policy: nothing big in $HOME) ----
export HF_HOME=$SCRATCH/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCH/.cache
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
export OLLAMA_MODELS=$SCRATCH/ollama/models
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p "$HF_HOME" "$OLLAMA_MODELS"

# ---- Python env ----
module load python/3.12.1
source "$SCRATCH/venvs/med-self-pref/bin/activate"
export PY=python

# ---- API keys ----
set -a; source .env; set +a

# ---- Start Ollama, wait for readiness, pull + alias the Qwen model ----
module load ollama/0.18.0
ollama serve > "multi_turn_zara_run/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

echo "Waiting for Ollama to come up..."
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
  || { echo "ERROR: Ollama did not start"; exit 1; }

echo "Pulling $QWEN_PULL ..."
ollama pull "$QWEN_PULL"        || { echo "ERROR: ollama pull failed"; exit 1; }
ollama cp "$QWEN_PULL" "$QWEN_ALIAS" 2>/dev/null || true
echo "Qwen served as alias: $QWEN_ALIAS"
export GEN_QWEN="$QWEN_ALIAS"   # run_4models.sh reads this as the 4th model id

# ---- Run the pipeline ----
if [ "${SMOKE:-0}" = "1" ]; then
  # Small on the expensive axis (2 scenarios) but keep the REAL turn design (gen 8t ->
  # truncate to 2/4/6/8) so every turn length is actually exercised.
  echo "=== SMOKE TEST: 2 scenarios, 8 gen-turns (evals 2/4/6/8t), 1000 tokens ==="
  bash multi_turn_zara_run/run_4models.sh 2 8 1000
else
  bash multi_turn_zara_run/run_4models.sh    # full run: 125 scenarios, 8 turns, 500 tokens
fi

echo "=== SBATCH done $(date) ==="
