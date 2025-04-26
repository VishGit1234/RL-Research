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
  vec_env = VecNormalize(make_vec_env(MujocoEnv, n_envs=20, env_kwargs={"cfg": cfg}))

  model = PPO('MlpPolicy', vec_env, verbose=1, policy_kwargs={"net_arch" : [256, 256]})
  model.learn(total_timesteps=1000000, progress_bar=False)

  model.save("ppo_model")

  del model # remove to demonstrate saving and loading

  model = PPO.load("ppo_model")

  cfg.viewer = False
  vec_env = VecNormalize(make_vec_env(MujocoEnv, n_envs=1, env_kwargs={"cfg": cfg}))
  obs = vec_env.reset()
  rewards = []
  for i in range(10000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      print(reward[0])
      rewards.append(reward[0])
      # VecEnv resets automatically
      # if done:
      #   obs = vec_env.reset()
  print(sum(rewards) / len(rewards))
if __name__ == '__main__':
  baseline("config.yaml")
