#!/bin/bash
#SBATCH --job-name=gr00t-calib
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -eu
module load FFmpeg/7.1.2
export PATH=$HOME/.local/bin:$PATH

# Venv convention: all venvs live under $WORK/venv/<name> (GPFS -- suited to
# the tens of thousands of small files a venv is; $HPCWORK is Lustre, tuned
# for few-large-file bandwidth and used for the dataset/checkpoints/caches
# instead).
VENV=$WORK/venv/finetune-groot
export PATH="$VENV/bin:$PATH"

# HF/torch caches -- these are set in ~/.zshrc but Slurm batch shells never
# source that (non-interactive bash), so export explicitly here.
export HF_HOME=$HPCWORK/hf-cache
export TORCH_HOME=$HPCWORK/torch-cache

# WANDB_API_KEY / HF_TOKEN must already be exported in the submitting shell
# environment -- never hardcoded here.
: "${WANDB_API_KEY:?WANDB_API_KEY not set in environment}"

CALIB_DIR=$HPCWORK/gr00t-ft/checkpoints/calibration_$SLURM_JOB_ID
mkdir -p "$CALIB_DIR"

cd ~/finetune-groot
finetune-groot \
    --config configs/finetune.yaml \
    --training.max_steps 200 \
    --training.save_steps 200 \
    --training.save_total_limit 1 \
    --training.output_dir "$CALIB_DIR" \
    --training.experiment_name "calib_${SLURM_JOB_ID}"

echo "=== calibration run complete ==="
ls -la "$CALIB_DIR"
