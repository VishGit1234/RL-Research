from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from KinovaGen3 import KinovaGen3

@register_env("KinovaPushCube", max_episode_steps=50)
class KinovaPushCubeEnv(PushCubeEnv):
    SUPPORTED_ROBOTS = [
        "kinova_gen3",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
