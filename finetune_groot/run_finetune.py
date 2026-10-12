"""training_cfgs-driven launcher around Isaac-GR00T's own launch_finetune.py.

configs/finetune.yaml carries both halves of a run (model + training).
training_cfgs.Config loads it and gives every field a typed
`--<group>.<field>` CLI override; `cfg.flatten()` then renders the same
values as the flat, kebab-case argv that NVIDIA's tyro-based
launch_finetune.py expects. The tested trainer path therefore runs
completely unmodified -- only its inputs come from our YAML.

Usage (`finetune-groot` is the console script declared in pyproject.toml;
prefix with `uv run` outside an activated venv):
    finetune-groot --config configs/finetune.yaml
    finetune-groot --config configs/finetune.yaml --training.max_steps 200
    finetune-groot --config configs/finetune.yaml --model.tune_llm true
"""

import subprocess
import sys
from pathlib import Path

from training_cfgs import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = REPO_ROOT / "vendor" / "Isaac-GR00T" / "gr00t" / "experiment" / "launch_finetune.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "finetune.yaml"


def main() -> None:
    cfg = Config.from_cli(default_config=DEFAULT_CONFIG, description="GR00T fine-tune launcher")
    print(cfg)

    cmd = [sys.executable, str(LAUNCH_SCRIPT), *cfg.flatten()]
    print("Launching:", " ".join(cmd))

    # WANDB_API_KEY / HF_TOKEN are inherited from the calling environment,
    # never read from this script or the yaml -- see calibrate_c23g.sbatch.
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
