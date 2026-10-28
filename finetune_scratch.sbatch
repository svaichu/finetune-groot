#!/bin/bash
#SBATCH --job-name=gr00t-scratch
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Run every cell of finetune_scratch.ipynb on one H100.
#   sbatch finetune_scratch.sbatch [RUN_CONFIG]     # default: configs/finetune_stack_tact_v4.yaml
# Executed notebook (with outputs): $HPCWORK/gr00t-ft/notebook-runs/finetune_scratch_<jobid>.ipynb
# (written only if every cell succeeds; the traceback of a failed run is in the Slurm log).
# Needs WANDB_API_KEY in your login environment and nbclient in the venv.

set -eu
unset PYTHONPATH                    # the module system leaves stale packages on it
module load FFmpeg/7.1.2            # torchcodec needs FFmpeg 4-7 at runtime
source $WORK/venv/finetune-groot/bin/activate
export HF_HOME=$HPCWORK/hf-cache TORCH_HOME=$HPCWORK/torch-cache
: "${WANDB_API_KEY:?WANDB_API_KEY not set in the login environment}"

cd ~/finetune-groot
export FINETUNE_REPO=$PWD
export FINETUNE_CONFIG=$(realpath "${1:-configs/finetune_stack_tact_v4.yaml}")

mkdir -p $HPCWORK/gr00t-ft/notebook-runs
jupyter execute finetune_scratch.ipynb --output=$HPCWORK/gr00t-ft/notebook-runs/finetune_scratch_$SLURM_JOB_ID
