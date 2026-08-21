#!/usr/bin/env bash
#SBATCH --job-name=stacked-hmbert-hipe
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out

# ETH Euler job skeleton. Module names drift between cluster software stacks
# -- run `module avail python` / `module avail cuda` and adjust before use.
# This assumes the `ner` conda environment (see NER/README.md) already has
# torch+CUDA, transformers, datasets, seqeval, pyyaml installed; if you
# instead manage env via `module load`, swap the two lines below.

module load stack/2024-06 python_cuda/3.11.6 2>/dev/null || true
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ner

mkdir -p logs
cd "$(dirname "$0")/.."

python scripts/build_dataset.py --config configs/default.yaml
python -m hipe_ner.train --config configs/default.yaml training.device=cuda
