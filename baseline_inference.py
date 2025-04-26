import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath("./tdmpc2"))
from tdmpc2.envs.mujoco_env import MujocoEnv
from tdmpc2.common.parser import parse_cfg

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

@parse_cfg
def baseline(cfg):
  model = PPO.load("ppo_model")

  cfg.viewer = True
  vec_env = VecNormalize(make_vec_env(MujocoEnv, n_envs=1, env_kwargs={"cfg": cfg}))
  obs = vec_env.reset()
  for i in range(10000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      # VecEnv resets automatically
      # if done:
      #   obs = vec_env.reset()

if __name__ == '__main__':
  baseline("config.yaml")
