#!/bin/bash
#SBATCH --job-name=sim_job
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=0-49
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu100

source ~/.bashrc
cleanenv
pyenv activate multiview

python eval_numerical.py --batch_num $SLURM_ARRAY_TASK_ID "$@"
