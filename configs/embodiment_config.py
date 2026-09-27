"""Embodiment/modality config for LSY-lab/stack_without_ft_tact_v4.

Registers EmbodimentTag.NEW_EMBODIMENT for GR00T-N1.7 fine-tuning, following
the schema in gr00t/configs/data/embodiment_configs.py. EmbodimentTag is a
fixed enum (see gr00t/data/embodiment_tags.py) -- custom embodiments must
register under NEW_EMBODIMENT ("new_embodiment"), not an arbitrary string;
--embodiment-tag on the CLI only accepts EmbodimentTag members.

Action representation was determined empirically (see
outputs/data_stats.csv / action_vs_state.png / gr00t_contract.md), not
assumed:
  - EEF (dims 0:6, x/y/z/roll/pitch/yaw) is RELATIVE -- action[t] matches the
    state delta cartesian[t+1] - cartesian[t] (RMSE ~0.002-0.013), not the
    absolute next state (RMSE ~0.07-3.1).
  - Gripper (dim 6) is ABSOLUTE and continuous (range [0, 0.999], not binary).
  - Real dataset fps is 15 (meta/info.json's outer video_info.fps=30 does not
    match; decoded frame count == parquet row count at fps=15).
  - Single task: "pick the lego block."

Horizon: 16 steps at 15 fps = ~1.07s, GR00T's standard default action-chunk
window; matches this dataset's fps (see gr00t_contract.md).
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
        modality_keys=["primary", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cartesian", "gripper", "joints", "target"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["eef", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                # ActionFormat.DEFAULT expects a 4x4 homogeneous SE(3) matrix, not
                # a 6-vector. Of the 6-dim options (XYZ_ROT6D needs 9, XYZ_ROTVEC
                # needs 6), only XYZ_ROTVEC matches our shape. The dataset's
                # roll/pitch/yaw columns are Euler angles, not literally an
                # axis-angle rotation vector, but per-step RELATIVE deltas are
                # tiny (max ~3.2 degrees total, see outputs/data_stats.csv) --
                # at that scale rotvec and Euler-difference are numerically
                # equivalent to first order, so this is a safe substitution for
                # the *delta* action, though NOT interchangeable for large
                # absolute rotations.
                format=ActionFormat.XYZ_ROTVEC,
                state_key="cartesian",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        # Must be prefixed "annotation." -- lerobot_episode_loader strips this
        # prefix and looks up the remainder in modality.json's "annotation"
        # section (see gr00t/data/dataset/lerobot_episode_loader.py:376).
        modality_keys=["annotation.human.task_description"],
    ),
}
