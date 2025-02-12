from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer

from timeit import default_timer

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, eval_env, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.eval_env = eval_env

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.eval_env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.eval_env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				# tim = default_timer()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				# print(default_timer() - tim)
				obs, reward, done, info = self.eval_env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.eval_env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.eval_env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, [True]*self.cfg.num_envs, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if any(done):
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					self._ep_idx = self.buffer.add(torch.cat(self._tds[self.cfg.num_envs:]))

				obs = self.env.reset()
				self._tds = [self.to_td(o) for o in obs]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(torch.stack(obs), t0=len(self._tds)==self.cfg.num_envs)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.extend([self.to_td(o, a, r) for o, a, r in zip(obs, action, reward)])

			# Update agent
			if self._step >= self.cfg.seed_steps and self._step % self.cfg.update_freq == 0:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_updates # self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = self.cfg.updates
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)
				# self.logger.log(train_metrics, 'train')

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)
