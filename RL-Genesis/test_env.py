import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import os, pickle
from kinova_env import KinovaEnv
from kinova_env_opt import KinovaEnvOpt
from config import env_cfg, train_cfg, MAX_ITERATIONS, Struct
import torch

def test():
    NUM_ENVS=3

    gs.init(backend=gs.gpu, logging_level="info")
    
    env = KinovaEnvOpt(
        num_envs=NUM_ENVS,
        env_cfg=Struct(**env_cfg),
        show_viewer=False
    )

    for i in range(10):
        obs = env.reset()
        done = torch.zeros(NUM_ENVS, dtype=torch.bool, device=obs.device)
        ep_reward = torch.zeros(NUM_ENVS, device=obs.device, dtype=gs.tc_float)
        while not done.any():
            action = obs[:, 2:4]/torch.norm(obs[:, 2:4], dim=1).unsqueeze(dim=1)
            action = action * torch.unsqueeze(obs[:, 1]/torch.norm(obs[:, :2], dim=1), dim=1)
            # action = torch.tensor([0,0], device=gs.device, dtype=gs.tc_float).unsqueeze(dim=0).repeat(NUM_ENVS, 1)
            obs, reward, done, info = env.step(actions=action)
            ep_reward += reward
            # print(torch.mean(reward).item())
        print("Episode success:", done.float().mean().item())
        print("Episode reward:", ep_reward.mean().item())

if __name__ == "__main__":
    test()
