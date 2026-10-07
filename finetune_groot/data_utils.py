"""Adapter between robotdataset batches and GR00T's VLAStepData/processor contract.

robotdataset's LeRobotVideoDataset returns TensorDict batches keyed by
slash-separated modality paths (e.g. "observation/images/primary"); GR00T's
own Gr00tN1d7Processor consumes a VLAStepData per sample, wrapped as a
one-element "messages" list. These helpers bridge the two without going
through gr00t's own LeRobotEpisodeLoader at all.

Usage::

    from finetune_groot.data_utils import (
        batch_to_vla_steps, load_dataset_statistics, steps_to_model_batch,
    )

    dataset_statistics = load_dataset_statistics(DATASET_PATH, modality_configs)
    processor.set_statistics({EMBODIMENT_TAG.value: dataset_statistics})

    batch = train_ds.sample()
    steps = batch_to_vla_steps(batch, embodiment_tag=EMBODIMENT_TAG)
    model_batch = steps_to_model_batch(steps, processor)
    out = model(**model_batch)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_dataset_statistics(dataset_path: Path, modality_configs: dict) -> dict:
    """Reproduce ``LeRobotEpisodeLoader.get_dataset_statistics()`` without the loader.

    Reads ``meta/stats.json`` (per raw-column mean/std/min/max/q01/q99) and
    ``meta/relative_stats.json`` (per action-state_key stats for RELATIVE
    action representations), slices each per the ``start``/``end`` indices
    in ``meta/modality.json``, and assembles the nested
    ``{modality: {joint_key: {stat_type: values}}}`` dict that
    ``Gr00tN1d7Processor.set_statistics`` expects (plus a top-level
    ``relative_action`` entry, passed through unsliced).

    Args:
        dataset_path: Root of the LeRobot v2.1 dataset tree (holds ``meta/``).
        modality_configs: The embodiment's ``{modality: ModalityConfig}`` dict
            (e.g. ``MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]``).

    Returns:
        Dict ready to pass as ``processor.set_statistics({tag: <this>})``.
    """
    meta_dir = Path(dataset_path) / "meta"
    with open(meta_dir / "stats.json") as f:
        stats = json.load(f)

    rel_path = meta_dir / "relative_stats.json"
    if rel_path.exists():
        with open(rel_path) as f:
            relative_stats = json.load(f)
        relative_stats.pop("__fingerprints__", None)
        stats["relative_action"] = relative_stats

    with open(meta_dir / "modality.json") as f:
        modality_meta = json.load(f)

    mapping = {"state": "observation.state", "action": "action"}
    out: dict = {}
    for modality in ("state", "action"):
        out[modality] = {}
        for joint_key in modality_configs[modality].modality_keys:
            meta = modality_meta[modality][joint_key]
            stats_key = meta.get("original_key") or mapping[modality]
            start, end = meta["start"], meta["end"]
            out[modality][joint_key] = {
                stat_type: values[start:end] for stat_type, values in stats[stats_key].items()
            }
    if "relative_action" in stats:
        out["relative_action"] = stats["relative_action"]
    return out


def batch_to_vla_steps(batch: Any, embodiment_tag: Any) -> list:
    """Convert one robotdataset TensorDict batch into a list of VLAStepData.

    Assumes the embodiment config registers video/state at anchor-only
    (``delta_indices=[0]``, i.e. T=1 in the batch) and action over a
    multi-step horizon. The dataset's raw 7-dim ``action`` column (EEF pose
    dims 0:6 + gripper dim 6) is sliced here into the two sub-modality keys
    ("eef", "gripper") that ``configs/embodiment_config.py`` registers --
    robotdataset does not know about this split since the source parquet
    stores one combined column.

    Args:
        batch: A TensorDict sampled from ``LeRobotVideoDataset``, with keys
            "action", "observation/images/{primary,wrist}",
            "observation/state/{cartesian,gripper,joints,target}", and
            optionally "task".
        embodiment_tag: The ``EmbodimentTag`` to stamp on each VLAStepData
            (e.g. ``EmbodimentTag.NEW_EMBODIMENT``).

    Returns:
        One ``gr00t.data.types.VLAStepData`` per sample in the batch.
    """
    from gr00t.data.types import VLAStepData

    b = batch.batch_size[0]
    steps = []
    action = batch["action"].numpy()  # (B, horizon, 7)
    primary = batch["observation/images/primary"].numpy()  # (B, 1, C, H, W)
    wrist = batch["observation/images/wrist"].numpy()
    cartesian = batch["observation/state/cartesian"].numpy()  # (B, 1, 6)
    gripper_s = batch["observation/state/gripper"].numpy()
    joints = batch["observation/state/joints"].numpy()
    target = batch["observation/state/target"].numpy()
    task = batch.get("task", None)

    for i in range(b):
        images = {
            "primary": [np.transpose(primary[i, 0], (1, 2, 0))],  # CHW -> HWC
            "wrist": [np.transpose(wrist[i, 0], (1, 2, 0))],
        }
        states = {
            "cartesian": cartesian[i],
            "gripper": gripper_s[i].reshape(1, -1) if gripper_s[i].ndim == 1 else gripper_s[i],
            "joints": joints[i],
            "target": target[i],
        }
        actions = {
            "eef": action[i, :, :6],
            "gripper": action[i, :, 6:7],
        }
        text = str(task[i]) if task is not None else "pick the lego block."
        steps.append(
            VLAStepData(
                images=images,
                states=states,
                actions=actions,
                text=text,
                embodiment=embodiment_tag,
            )
        )
    return steps


def steps_to_model_batch(steps: list, processor: Any) -> dict:
    """Run each VLAStepData through ``processor`` and collate into one model batch."""
    from gr00t.data.types import MessageType

    features = [
        processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]) for step in steps
    ]
    return processor.collator(features)
