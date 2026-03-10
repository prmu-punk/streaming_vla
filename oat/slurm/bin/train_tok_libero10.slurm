#!/bin/bash
#SBATCH --job-name="bintok"
#SBATCH --output=...
#SBATCH --error=...
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --time=00-48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=...


# navigate to the workspace
cd /path/to/project/oat

HYDRA_FULL_ERROR=1 uv run python \
    scripts/run_workspace.py \
    --config-name=train_bintok \
    task/tokenizer=libero/libero10 \
    training.num_demo=500
