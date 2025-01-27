from copy import deepcopy
import warnings

import gymnasium

from envs.mujoco_env import MujocoEnv

from envs.wrappers.tensor import TensorWrapper

	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gymnasium.logger.min_level = 40
	env = MujocoEnv(cfg=cfg)

	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
	env = TensorWrapper(env)
	# if cfg.get('obs', 'state') == 'rgb':
	# 	env = PixelWrapper(cfg, env)
	cfg.obs_shape = {'state': env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
