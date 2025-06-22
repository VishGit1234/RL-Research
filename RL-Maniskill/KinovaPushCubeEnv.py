from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from KinovaGen3 import KinovaGen3
import torch

@register_env("KinovaPushCube", max_episode_steps=50)
class KinovaPushCubeEnv(PushCubeEnv):
    SUPPORTED_ROBOTS = [
        "kinova_gen3",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # set the keyframe for the robot
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
