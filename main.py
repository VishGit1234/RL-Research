import random
from mujoco_env import MujocoEnv
import numpy as np

env = MujocoEnv()
env.reset()

for i in range(10000):
  o, r, done, truncated, info = env.step(np.array([1 - random.random()*2 for i in range(7)]))
