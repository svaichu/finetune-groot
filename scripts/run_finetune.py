"""training_cfgs-driven launcher around Isaac-GR00T's own launch_finetune.py.

Loads hyperparameters from a YAML file via training_cfgs.Config (typed
--group.field CLI overrides, same dotted keys training_cfgs' own wandb-sweep
export uses), then invokes NVIDIA's launch_finetune.py as a subprocess with
the equivalent CLI flags -- the tested trainer path is reused completely
unmodified, only its inputs are supplied by training_cfgs instead of tyro's
default CLI experience.

Usage:
    python run_finetune.py --config configs/finetune.yaml
    python run_finetune.py --config configs/finetune.yaml --training.max_steps 200
    python run_finetune.py --config configs/finetune.yaml --training.max_steps 200 --paths.output_dir /tmp/calib
"""

import os
import subprocess
import sys
from pathlib import Path

from training_cfgs import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = REPO_ROOT / "vendor" / "Isaac-GR00T" / "gr00t" / "experiment" / "launch_finetune.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "finetune.yaml"

# FinetuneConfig field -> (group, field) in our YAML.
FIELD_MAP = {
    "base-model-path": ("paths", "base_model_path"),
    "dataset-path": ("paths", "dataset_path"),
    "embodiment-tag": ("paths", "embodiment_tag"),
    "modality-config-path": ("paths", "modality_config_path"),
    "output-dir": ("paths", "output_dir"),
    "tune-llm": ("tuning", "tune_llm"),
    "tune-visual": ("tuning", "tune_visual"),
    "tune-projector": ("tuning", "tune_projector"),
    "tune-diffusion-model": ("tuning", "tune_diffusion_model"),
    "global-batch-size": ("training", "global_batch_size"),
    "dataloader-num-workers": ("training", "dataloader_num_workers"),
    "learning-rate": ("training", "learning_rate"),
    "gradient-accumulation-steps": ("training", "gradient_accumulation_steps"),
    "save-steps": ("training", "save_steps"),
    "save-total-limit": ("training", "save_total_limit"),
    "num-gpus": ("training", "num_gpus"),
    "max-steps": ("training", "max_steps"),
    "warmup-ratio": ("training", "warmup_ratio"),
    "use-wandb": ("logging", "use_wandb"),
    "wandb-project": ("logging", "wandb_project"),
    "experiment-name": ("logging", "experiment_name"),
}

# tyro renders bool fields as --flag / --no-flag, not --flag true/false.
BOOL_FLAGS = {"tune-llm", "tune-visual", "tune-projector", "tune-diffusion-model", "use-wandb"}


def build_argv(cfg: Config) -> list[str]:
    argv = []
    for cli_name, (group, field) in FIELD_MAP.items():
        value = getattr(getattr(cfg, group), field)
        if value is None:
            continue
        if cli_name in BOOL_FLAGS:
            argv.append(f"--{cli_name}" if value else f"--no-{cli_name}")
        else:
            argv.extend([f"--{cli_name}", str(value)])
    return argv


def main() -> None:
    cfg = Config.from_cli(default_config=DEFAULT_CONFIG, description="GR00T fine-tune launcher")
    print(cfg)

    argv = build_argv(cfg)
    cmd = [sys.executable, str(LAUNCH_SCRIPT), *argv]
    print("Launching:", " ".join(cmd))

    # WANDB_API_KEY / HF_TOKEN come from the calling environment, never from
    # this script or the yaml -- see slurm/calibrate_c23g.sbatch.
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
