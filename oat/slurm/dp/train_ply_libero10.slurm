#!/bin/bash
#SBATCH --job-name="difply"
#SBATCH --output=...
#SBATCH --error=...
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --time=00-48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=...


# navigate to the workspace
cd /path/to/project/oat

# launch job
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines 1 \
    --multi_gpu \
    --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_diffpolicy \
    task/policy=libero/libero10 \
    training.num_epochs=5001 \
    training.num_demo=500 \
    task.policy.lazy_eval=false
