from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
import numpy as np
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
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.05, -0.05] to [0.05, 0.05]
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.1 - 0.05
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even use env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
            # and we further rotate 90 degrees on the y-axis to make the target object face up
            target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            # set the keyframe for the robot
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
