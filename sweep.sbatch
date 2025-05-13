#!/bin/bash
#SBATCH --job-name=hyper
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=00:30:00            # adjust
#SBATCH --gres=gpu:1               # or omit/modify
#SBATCH --array=0-2399%32   # 2 400 tasks, max 32 running in parallel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-$(($(wc -l < grid.txt) - 1))

# ---------- core ----------
set -e

# 1-based → 0-based line number
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" grid.txt)
echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] running: $CMD"

module load cuda/12.2            # if you need modules
source ~/envs/myenv/bin/activate # or conda activate …

python train.py \
  --model-dir results/numerical \
  --model-id run_${SLURM_ARRAY_TASK_ID} \
  $CMD
