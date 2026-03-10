#!/bin/bash
#SBATCH --job-name="oattok"
#SBATCH --output=...
#SBATCH --error=...
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --time=00-48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=...


# navigate to the workspace
cd /path/to/project/oat

# launch job
HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines 1 \
    --multi_gpu \
    --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=5001 \
    training.num_demo=500