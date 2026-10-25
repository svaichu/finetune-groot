"""Embodiment/modality config for ElRobot (norma-core / Station).

Registers EmbodimentTag.NEW_EMBODIMENT for GR00T-N1.7, following the schema
in gr00t/configs/data/embodiment_configs.py. EmbodimentTag is a fixed enum
(see gr00t/data/embodiment_tags.py) -- custom embodiments must register under
NEW_EMBODIMENT ("new_embodiment"), not an arbitrary string; --embodiment-tag
on the CLI only accepts EmbodimentTag members.

Shape/representation came from inference-client (DATA_FORMAT.md,
src/inference_client/frame.py, notebooks/parquet_to_obs.ipynb) and
dataset-cheese1.parquet, not assumption:
  - 8 joints, 3 cameras, one flat vector each -- Station's `normvla.Frame`
    has no EEF/gripper split, so this is plain joint-space, not
    EEF+gripper like the old LSY-lab config.
  - State = each joint's `position_norm` (measured position, [0,1] over
    (range_min, range_max)), in Frame order.
  - Action = each joint's `goal_norm` (commanded position, same [0,1]
    scale). Checked empirically against both hypotheses on
    dataset-cheese1.parquet (408 steps, 8 joints): RMSE(goal[t], pos[t+1])
    ~0.001-0.054 per joint vs RMSE(goal[t], pos[t]) ~0.001-0.067 -- close
    enough (low-speed teleop) that neither next-state nor same-step framing
    is clearly better, so this is ABSOLUTE (an absolute commanded target),
    not a RELATIVE delta.
  - fps ~10 (median inter-frame dt ~100ms in dataset-cheese1.parquet).
  - 3 cameras, no name info beyond index -- inference-client hardcodes
    "cam0"/"cam1"/"cam2" in Frame.images order (DATA_FORMAT.md, notebook);
    matched here so a served checkpoint's modality config lines up with
    what the bridge sends without extra translation.
  - Language/state key names ("task" / "state") are literal lookups, not
    cosmetic: Gr00tPolicy resolves observation["language"][language_key]
    and observation["state"][state_key] directly (gr00t_policy.py), and
    StationFrame.to_observation() builds exactly {"task": ...} / {"state":
    ...} -- so these names must match the bridge's output, not a
    lerobot "annotation.*" convention (that prefix is specific to
    lerobot_episode_loader, which ElRobot's data doesn't go through).

Horizon: 16 steps, GR00T's standard default action-chunk window (~1.6s at
~10 fps here, vs ~1.07s at 15 fps for the old LSY-lab config).

Training: this file registers the model-facing shape for the legacy
launch_finetune.py flow (--modality-config-path). For the newer
robotdataset-adapter flow (finetune_scratch.ipynb), the same shape is
described as data by configs/modalities/elrobot.modality.yaml, read via
robotdataset.station_dataset.StationParquetDataset (one .parquet file per
episode -- see that module's docstring in /workspace/robotdataset) and
robotdataset.adapters.gr00t.Gr00tAdapter (existing, dataset-agnostic;
nothing new needed there). Keep the two in sync: the elrobot.modality.yaml
file is checked against this one in finetune-groot's own test suite the
same way stack_tact_v4.modality.yaml is checked against the old LSY-lab
config.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value] = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam0", "cam1", "cam2"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}
