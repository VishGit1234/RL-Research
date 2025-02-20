from .tdmpc2 import TDMPC2
from common.parser import parse_cfg
from envs import make_env
import os
from timeit import default_timer

@parse_cfg
def inference(cfg):
	cfg.viewer = True
	cfg.mpc = False
	env = make_env(cfg)
	model = TDMPC2(cfg)
	model.load(os.path.join(os.path.abspath('log'), 'logs', 'models', 'final.pt'))
	for i in range(10000):
		obs = env.reset()
		done, reward = False, 0
		ep_reward, t = 0, 0  # Initialize ep_reward and t
		while not done:
			# tim = default_timer()
			action = model.act(obs, t0=t==0, eval_mode=True)
			# print(default_timer() - tim)
			obs, reward, done, info = env.step(action)
			print(reward)
			ep_reward += reward
			t += 1

