#!/usr/bin/env bash
#SBATCH --job-name=stacked-hmbert-hipe
#SBATCH --partition=gpu.24h
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ETH Euler job for hipe_ner training. Module names drift between cluster
# software-stack releases -- run `module avail python_cuda` on a login node
# and adjust the `module load` line below if this fails with "module not
# found".
#
# --- One-time setup required BEFORE submitting this job (on a LOGIN node,
#     which has internet; COMPUTE nodes do not) ---
#   1. venv created and this package installed into it:
#        module load stack/2024-06 python_cuda/3.11.6
#        python -m venv ~/ner-venv
#        source ~/ner-venv/bin/activate
#        pip install -e .
#   2. HIPE data is cloned:
#        scripts/download_hipe_data.sh
#   3. hmBERT weights are cached locally, so the compute node doesn't need
#      to reach huggingface.co:
#        python -c "from transformers import AutoTokenizer, AutoModel; \
#          m='dbmdz/bert-base-historic-multilingual-cased'; \
#          AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
# This script checks for (2) and (3) and fails fast with a clear message
# instead of hanging on a network call that will never succeed.

set -euo pipefail

env2lmod 2>/dev/null || true   # switch to the new software stack if not already on it
module load stack/2024-06 python_cuda/3.11.6

source ~/ner-venv/bin/activate

cd "$(dirname "$0")/.."
mkdir -p logs

if [ ! -d data/raw/data/v2.1 ]; then
    echo "ERROR: data/raw/data/v2.1 not found. Run scripts/download_hipe_data.sh on a login node first." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1     # fail fast instead of hanging: compute nodes have no internet
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

if ! python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('dbmdz/bert-base-historic-multilingual-cased')
" >/dev/null 2>&1; then
    echo "ERROR: hmBERT tokenizer not found in local HF cache ($HF_HOME)." >&2
    echo "Pre-download it on a login node (see comment block at top of this script)." >&2
    exit 1
fi

# $HOME has a small quota on Euler; write checkpoints to $SCRATCH instead
# (note: $SCRATCH is purged of files untouched for 15 days -- the copy-back
# step below moves the final checkpoints to the canonical repo path so
# scripts/evaluate.py and scripts/infer.py work unmodified, per the README).
# This path is deliberately NOT job-ID-specific: hipe_ner.train auto-resumes
# from <output_dir>/resume_state.pt if present, so a resubmitted job (e.g.
# after hitting the gpu.24h time limit) needs to land in the SAME scratch
# directory as the run it's continuing, not a fresh one.
FINAL_DIR="checkpoints/stacked-hmbert-hipe"
SCRATCH_OUT="${SCRATCH:?SCRATCH not set}/hipe_ner_checkpoints/stacked-hmbert-hipe"
mkdir -p "$SCRATCH_OUT"

CACHE_DIR="data/cache"
if [ -f "$CACHE_DIR/label_map.json" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
    echo "reusing existing $CACHE_DIR (set FORCE_REBUILD=1 to rebuild)"
else
    python scripts/build_dataset.py --config configs/default.yaml
fi

python -m hipe_ner.train --config configs/default.yaml \
    training.device=cuda \
    training.output_dir="$SCRATCH_OUT"

mkdir -p "$(dirname "$FINAL_DIR")"
rm -rf "$FINAL_DIR"
cp -r "$SCRATCH_OUT" "$FINAL_DIR"
echo "checkpoints copied back to $FINAL_DIR"
