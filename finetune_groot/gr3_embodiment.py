"""Embodiment/modality config for the simulated GR-3 pick-up env (gr3-humanoid, robotenv).

Registers EmbodimentTag.NEW_EMBODIMENT. Keys and order must match ``gr3-humanoid/gr3_embodiment.py`` (the env-side
``JointLayout``), which maps them onto robot joints at inference time.

  - state : right arm (7 joint angles, rad) + right gripper (2 finger slides, m), absolute, one frame.
  - action: same joints, ABSOLUTE joint position targets (the env runs in ``joint_pos_abs``), 16-step chunk.
  - video : rs_frontview ("front") and rs_agentview ("agent"), one frame.

Import this module before building the processor / starting the server so the config is registered.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 16

MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value] = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "agent"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["right_arm", "right_gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["right_arm", "right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
            for _ in range(2)
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
