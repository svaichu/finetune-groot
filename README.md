# finetune-groot

GR00T-N1.7 fine-tuning on [`robotdataset`](https://github.com/svaichu/robotdataset) datasets
(`LSY-lab/stack_without_ft_tact_v4`, Open X-Embodiment `viola`, ...), driven by `robotdataset` loaders and
[`training_cfgs`](https://pypi.org/project/training-cfgs/) YAML configs.
NVIDIA's trainer runs unmodified from `vendor/Isaac-GR00T`; this repo supplies the notebook, the run files and the
config plumbing, and `robotdataset` supplies the dataset handling (dataset info and the GR00T model adapter).

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

The two repos are expected to sit side by side (`robotdataset` 0.4.0 is not on
PyPI yet, and until it is merged into `devel` it lives on the branch `feature/model-adapters`):

```
<parent>/
  finetune-groot/      <- this repo, with vendor/Isaac-GR00T checked out
  robotdataset/        <- branch feature/model-adapters (0.4.0)
```

```bash
git -C ../robotdataset fetch origin feature/model-adapters && git -C ../robotdataset switch feature/model-adapters
```

## Fine-tuning on any `robotdataset` dataset

`finetune_scratch.ipynb` trains GR00T-N1.7 on whatever `robotdataset` can load (`LeRobotVideoDataset`,
`OXEDataset`, `Table30v2Dataset`, ...). The notebook never opens a dataset's `meta/*.json`; the dataset is only
handled by `robotdataset`, in two layers that live in that package:

1. **Dataset info** (`robotdataset.info`) lists every modality a loader provides as an editable YAML.
2. **Model adapter** (`robotdataset.adapters.gr00t.Gr00tAdapter`) reads the edited YAML, builds the loader with the
   sampling windows the selection needs, produces the GR00T `ModalityConfig`s and `VLAStepData`, and computes the
   normalisation statistics the processor needs (from the training loader, in memory).

To train on a new dataset:

```bash
# 1. list the modalities (a few episodes are enough) and write the default YAML
python -m robotdataset.info --loader OXEDataset --kwargs '{"dataset_name": "viola"}' \
    --episodes 0:8 --adapter gr00t -o configs/scratch/my_dataset.modality.yaml

# 2. edit it: `use: true` on the modalities the model should get, their order, names, `split` for a vector
#    leaf that packs several groups, and the action representation in the `gr00t:` block
# 3. copy configs/scratch/viola.yaml to configs/scratch/my_dataset.yaml; set data.modality_file, the train and
#    validation episodes and the training hyperparameters
# 4. run the notebook on that run file
FINETUNE_CONFIG=configs/scratch/my_dataset.yaml jupyter lab finetune_scratch.ipynb
```

| file | holds |
|---|---|
| `configs/scratch/<name>.yaml` | run file, read with `training_cfgs.Config`: model, training, `data.train_episodes` / `data.val_episodes` (ints or `"a:b"` ranges), `data.modality_file` |
| `configs/scratch/<name>.modality.yaml` | dataset-info file: loader and arguments, modalities picked and ordered, action representation, windows |

Shipped examples: `stack_tact_v4` (`LeRobotVideoDataset`, EEF action `RELATIVE` to the state `cartesian`,
gripper `ABSOLUTE`) and `viola` (`OXEDataset`, two cameras, joint + gripper state, delta-command actions, language
from the dataset). Statistics and the position of every group inside the assembled state and action vectors come
from the adapter (`adapter.index_map()`); each checkpoint also receives the modality file the run used
(`dataset_info.yaml`) and a copy of the run file.

`OXEDataset` needs `robotdataset[oxe]`. Cached episodes load without internet access; missing ones are streamed
from `gs://gresearch/robotics` on first use. On the RWTH cluster robotdataset points TensorFlow's `gs://` client at
the system CA bundle itself; set `CURL_CA_BUNDLE` to override (see robotdataset `doc/datasets/oxe.md`).

`finetune_groot/embodiment_config.py`, `run_finetune.py` and the `finetune*.yaml` configs belong to the vendored
`launch_finetune.py` flow and are unchanged.

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
!git clone --branch feature/model-adapters https://github.com/svaichu/robotdataset.git /content/robotdataset
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
(`pip install -e vendor/Isaac-GR00T -e ../robotdataset[hf,oxe] -e .`), but you must
install the flash-attn wheel URL from `[tool.uv.sources]` first: pip does not
read that table, and flash-attn is sdist-only on PyPI (no binary wheel for any
version), so pip would spend ~1 h compiling CUDA kernels.
