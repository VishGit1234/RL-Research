from KinovaPushCubeEnv import KinovaPushCubeEnv
import gymnasium as gym
import torch

env = gym.make("KinovaPushCube", render_mode="human", robot_uids="kinova_gen3", max_episode_steps=50, control_mode="pd_ee_delta_pose")
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
terminated = torch.zeros(1, dtype=torch.bool)
truncated = torch.zeros(1, dtype=torch.bool)
while not torch.any(terminated) or torch.any(truncated):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
print(obs, rew, terminated, truncated, info)