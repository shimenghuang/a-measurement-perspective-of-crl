#!/bin/bash
#SBATCH --job-name=sim_job
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu100

source ~/.bashrc
cleanenv
pyenv activate multiview

python run_three_latent_nonlinear.py "$@"
