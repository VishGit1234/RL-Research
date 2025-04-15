from .tdmpc2 import TDMPC2
from common.parser import parse_cfg
from envs import make_env
import os
from timeit import default_timer
from time import sleep
import numpy as np

@parse_cfg
def inference(cfg):
	cfg.viewer = True
	# cfg.mpc = False
	env = make_env(cfg)
	# model = TDMPC2(cfg)
	# model.load(os.path.join(os.path.abspath('log'), 'logs', 'models', 'final.pt'))
	for i in range(300):
		obs = env.reset()
		done, reward = False, 0
		ep_reward, t = 0, 0  # Initialize ep_reward and t
		for i in range(cfg.max_episode_steps):
			# tim = default_timer()
			# action = model.act(obs, t0=t==0, eval_mode=True)
			# env.env.goal = [0.4, 0.35, 0.02]
			action = env.rand_act()
			# print(default_timer() - tim)
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			t += 1
			sleep(0.01)
		print(ep_reward)

