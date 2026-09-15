from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionType, ActionRepresentation, ActionFormat
from gr00t.data.types import EmbodimentTag


my_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "observation.images.primary",
            "observation.images.wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "observation.state.cartesian",
            "observation.state.gripper",
            "observation.state.joints",
            "observation.state.target"
        ],
        action_configs=[
            # cartesian
            ActionConfig(
                rep=ActionRepresentation.RELATIVE, # TODO: lookup doc of dataset
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # joints
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # target
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE, #TODO: lookup doc of dataset
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
        ]
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 7)),
        modality_keys=[
            "action"
        ],
        action_configs=[
            # single_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper
            # ActionConfig(
            #     rep=ActionRepresentation.ABSOLUTE,
            #     type=ActionType.NON_EEF,
            #     format=ActionFormat.DEFAULT,
            # ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(my_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)