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

VENV=$HPCWORK/gr00t-ft/venvs/gr00t
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
python scripts/run_finetune.py \
    --config configs/finetune.yaml \
    --training.max_steps 200 \
    --training.save_steps 200 \
    --training.save_total_limit 1 \
    --paths.output_dir "$CALIB_DIR" \
    --logging.experiment_name "calib_${SLURM_JOB_ID}"

echo "=== calibration run complete ==="
ls -la "$CALIB_DIR"
