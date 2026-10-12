# finetune-groot

GR00T-N1.7 fine-tuning on `LSY-lab/stack_without_ft_tact_v4`, driven by
[`robotdataset`](https://github.com/svaichu/robotdataset) loaders and
[`training_cfgs`](https://pypi.org/project/training-cfgs/) YAML configs.
NVIDIA's trainer runs unmodified from `vendor/Isaac-GR00T`; this repo only
supplies the dataset adapter, the modality spec and the config plumbing.

## Version contract

Everything heavy is pinned **once**, by the vendored release:

| | source of truth |
|---|---|
| Python `==3.10.*`, torch 2.7.1, torchvision, torchcodec 0.4.0, transformers 4.57.3, numpy 1.26.4, flash-attn 2.7.4.post1, wandb | `vendor/Isaac-GR00T/pyproject.toml` (tag `n1.7-release`) |
| `robotdataset`, `training-cfgs` | this `pyproject.toml` |

`pyproject.toml` here deliberately does **not** repeat those pins. If you bump
the vendored checkout, re-read its `requires-python` and update only
`requires-python` here to match — nothing else.

## Layout

The two repos are expected to sit side by side (`robotdataset` 0.3.0 is not on
PyPI yet):

```
<parent>/
  finetune-groot/      <- this repo, with vendor/Isaac-GR00T checked out
  robotdataset/
```

## Install

One command, same everywhere — [`uv`](https://docs.astral.sh/uv/) fetches its
own CPython 3.10, so no system Python 3.10 or conda is required:

```bash
cd finetune-groot
uv sync                       # add --extra dev for pytest
```

`uv sync` resolves ~1-2 min and writes `uv.lock`; commit the lock so every
machine gets the identical environment. Run things with `uv run`:

```bash
uv run finetune-groot --config configs/finetune.yaml
uv run finetune-groot --config configs/finetune.yaml --training.max_steps 200
uv run python -c "import gr00t, torch; print(torch.__version__)"
```

`torchcodec` needs a system FFmpeg in the 4–7 range (8+ breaks it).

### RWTH cluster

The venv lives outside the repo, per the `$WORK/venv/<project>` convention:

```bash
module load FFmpeg/7.1.2
export UV_PROJECT_ENVIRONMENT=$WORK/venv/finetune-groot
export UV_CACHE_DIR=/tmp/$USER/uv-cache-ftg UV_LINK_MODE=copy   # GPFS is slow for uv's cache
unset PYTHONPATH
~/.local/bin/uv sync
```

Build it in a `c23ms` job, not on the login node (~5 min, ~10 GB; the login
shell's 60 s command limit will cut it off). Compute nodes reach pypi.org and
github.com, so the install works from inside the job. Activate later with
`source $WORK/venv/finetune-groot/bin/activate`.

### Google Colab

```python
!pip install -q uv
!git clone https://github.com/svaichu/robotdataset.git /content/robotdataset
!git clone --recurse-submodules <finetune-groot-url> /content/finetune-groot
!apt-get install -y -qq ffmpeg                      # Ubuntu 24.04 ships FFmpeg 6.1
!cd /content/finetune-groot && uv sync
!cd /content/finetune-groot && uv run finetune-groot --config configs/finetune.yaml
```

Colab's own kernel is Python 3.12 and cannot import `gr00t`, so drive training
through `uv run` (a subprocess in the 3.10 venv) rather than importing in the
notebook. On a T4 (sm75) also set `use_flash_attention: false` in the model
config — flash-attn needs Ampere or newer; gr00t then falls back to sdpa.

Plain `pip` works too if the interpreter already is 3.10
(`pip install -e vendor/Isaac-GR00T -e ../robotdataset[hf] -e .`), but you must
install the flash-attn wheel URL from `[tool.uv.sources]` first: pip does not
read that table, and flash-attn is sdist-only on PyPI (no binary wheel for any
version), so pip would spend ~1 h compiling CUDA kernels.
