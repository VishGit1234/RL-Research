from collections import defaultdict

import gymnasium
import numpy as np
import torch

class TensorVectorWrapper(gymnasium.vector.VectorWrapper):
	"""
	Vector env Wrapper for converting  numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None, **kwargs):
		return [self._obs_to_tensor(o) for o in self.env.reset()[0]]

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(np.mean(info['success']))
		o = [self._obs_to_tensor(o) for o in obs]
		r = [torch.tensor(r, dtype=torch.float32) for r in reward]
		return o, r, terminated, info

class TensorWrapper(gymnasium.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset()[0])

	def step(self, action):
		obs, reward, done, _, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
