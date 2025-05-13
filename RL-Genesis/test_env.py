from kinova_env import KinovaEnv
import genesis as gs
import numpy as np
import torch

def test():
    gs.init(backend=gs.gpu)

    NUM_ENVS=3

    env = KinovaEnv(
        num_envs=NUM_ENVS,
        env_cfg=dict(
            episode_length_s=10,
            init_joint_angles=np.array([6.25032076, 0.60241784, 3.15709118, -2.128586102, 6.28220792, -0.39995964788322566, 1.55241801, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822]),
            init_quat=np.array([1, 0, 0., 1]),
            bracelet_link_height=0.25,
            init_box_pos=(0.2, 0.2, 0.02),
            box_size=(0.08, 0.08, 0.02),
            clip_actions=0.01,
            termination_if_cube_goal_dist_less_than=0.001,
            cube_goal_dist_rew_scale=3,
            cube_arm_dist_rew_scale=2,
            success_reward=100,
            target_displacement=0.1
        ),
        show_viewer=True
    )

    env.reset()

    for i in range(1000):
        action = 2*0.01*(torch.rand((NUM_ENVS, 2)) - 0.5)
        env.step(actions=action)

if __name__ == "__main__":
    test()