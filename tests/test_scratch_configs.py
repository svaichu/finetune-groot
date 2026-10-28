"""The run files and modality files in configs/scratch load, and describe what the notebook trains.

No GPU, no dataset and no model are needed: the files are parsed with ``training_cfgs`` and
``robotdataset``'s adapter, and the notebook source is checked for what it must not do.
"""

from __future__ import annotations

import ast
import json
import re
import dataclasses
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "configs" / "scratch"
RUNS = sorted(SCRATCH.glob("*.yaml"))
RUNS = [p for p in RUNS if not p.name.endswith(".modality.yaml")]


def _plain(config):
    """A ModalityConfig as plain data, whether GR00T defines it as a dataclass or a pydantic model."""
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return config.model_dump()


def test_there_are_runs():
    assert {p.stem for p in RUNS} >= {"stack_tact_v4", "viola"}


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.stem)
def test_run_file_loads_and_points_at_an_existing_modality_file(run):
    from training_cfgs import Config

    pytest.importorskip("robotdataset.info")
    from robotdataset.info import parse_episodes

    cfg = Config.from_file(run)
    train, val = parse_episodes(cfg.data.train_episodes), parse_episodes(cfg.data.val_episodes)
    assert train and val and not set(train) & set(val)
    assert (REPO / cfg.data.modality_file).is_file(), cfg.data.modality_file
    for field in ("global_batch_size", "learning_rate", "max_steps", "warmup_ratio", "max_grad_norm", "save_steps", "logging_steps",
                  "output_dir", "experiment_name", "use_wandb", "wandb_project"):
        assert hasattr(cfg.training, field), field
    for field in ("base_model_path", "embodiment_tag", "tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model", "load_bf16"):
        assert hasattr(cfg.model, field), field


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.stem)
def test_modality_file_builds_an_adapter_without_the_dataset(run):
    from training_cfgs import Config

    pytest.importorskip("robotdataset.adapters.gr00t")
    from robotdataset.adapters.gr00t import Gr00tAdapter

    cfg = Config.from_file(run)
    adapter = Gr00tAdapter.from_yaml(REPO / cfg.data.modality_file)
    assert adapter.groups["video"] and adapter.groups["state"] and adapter.groups["action"]
    windows = adapter.delta_timestamps()
    assert all(len(v) == adapter.action_horizon for p, v in windows.items() if p in {g.path for g in adapter.groups["action"]})
    index = adapter.index_map()
    assert [e["group"] for e in index["action"]] == [g.name for g in adapter.groups["action"]]


def test_stack_tact_file_reproduces_the_legacy_embodiment_config():
    """configs/scratch/stack_tact_v4.modality.yaml == finetune_groot/embodiment_config.py (still used by the vendor flow)."""
    pytest.importorskip("gr00t.data.types")
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from robotdataset.adapters.gr00t import Gr00tAdapter

    import finetune_groot.embodiment_config  # noqa: F401  (registers NEW_EMBODIMENT)

    legacy = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    new = Gr00tAdapter.from_yaml(SCRATCH / "stack_tact_v4.modality.yaml").modality_configs()
    assert set(new) == set(legacy)
    for modality in legacy:
        assert _plain(new[modality]) == _plain(legacy[modality]), modality


def test_elrobot_file_builds_the_expected_modality_configs():
    """configs/modalities/elrobot.modality.yaml (adapter flow) -> cam0..2 / state / action(ABSOLUTE) / task."""
    pytest.importorskip("gr00t.data.types")
    from robotdataset.adapters.gr00t import Gr00tAdapter

    cfgs = Gr00tAdapter.from_yaml(REPO / "configs" / "modalities" / "elrobot.modality.yaml").modality_configs()
    assert cfgs["video"].modality_keys == ["cam0", "cam1", "cam2"]
    assert cfgs["state"].modality_keys == ["state"]
    assert cfgs["action"].modality_keys == ["action"]
    assert list(cfgs["action"].delta_indices) == list(range(16))
    assert cfgs["language"].modality_keys == ["task"]


def _code_cells(path: Path):
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_cells_parse_and_never_read_dataset_metadata():
    cells = _code_cells(REPO / "finetune_scratch.ipynb")
    assert cells
    for src in cells:
        ast.parse(src)
    code = "\n".join(cells)
    assert not re.search(r"info\.json|modality\.json|stats\.json|relative_stats|import json|json\.load", code)
    assert "finetune_groot" not in code  # no local glue: robotdataset's adapter does the conversion
    assert "Gr00tAdapter" in code and "configure_processor" in code and "to_model_batch" in code


def test_notebook_has_no_hardcoded_hyperparameters():
    code = "\n".join(_code_cells(REPO / "finetune_scratch.ipynb"))
    for token in ("1e-4", "1.0e-4", "batch_size=8", "max_steps=", "learning_rate=1", "delta_timestamps", "slice(", "[:6]", "[6:7]"):
        assert token not in code, token
