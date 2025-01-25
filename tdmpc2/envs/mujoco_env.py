import numpy as np
import os
import mujoco
import mujoco.viewer
import gymnasium
from gymnasium.spaces import Box
import random

class MujocoEnv(gymnasium.Env):
  def __init__(self, **kwargs):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kinova_gen3', 'scene.xml')
    self.model = mujoco.MjModel.from_xml_path(path)
    self.sim = mujoco.MjData(self.model)
    # goal should be the position we want the end of the arm to be at
    # should be an numpy array
    self.timestep = 0
    self.done = False

    self.goal = MujocoEnv._generate_goal()
    self.max_episode_steps = 1000000

    self.action_space = Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (7,), np.float32)
    self.observation_space = Box(-np.inf, np.inf, (17,), np.float32)

    self.viewer = mujoco.viewer.launch_passive(self.model, self.sim)
  
  def _generate_goal():
    MIN_RADIUS = 0.4
    MAX_RADIUS = 0.8

    return np.array([(random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                    (random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                    random.uniform(MIN_RADIUS, MAX_RADIUS)])

  def step(self, action):
    # print(self.goal)
    self.timestep += 1

    # Apply the action to the environment
    self.sim.ctrl[:] = action
    mujoco.mj_step(self.model, self.sim)

    # Get the observation, reward, done, and info
    observation = self._get_observation()
    reward = self._get_reward()
    # done = self._get_done()
    done = False
    self.done = done
    truncated = False
    info = {}

    # update viewer
    self.viewer.sync()

    if self.timestep > 1000:
      done = True
      self.done = True

    return observation, reward, done, info

  def reset(self):
    # Reset MuJoCo
    mujoco.mj_resetData(self.model, self.sim)
    mujoco.mj_forward(self.model, self.sim)

    self.timestep = 0
    self.done = False

    # Get observation 
    self.goal = MujocoEnv._generate_goal()
    mujoco.mj_forward(self.model, self.sim)
    
    obs = self._get_observation()
    reset_info = {}  # This can be populated with any reset-specific info if needed

    return obs

  def _get_observation(self):
    # Joint positions
    qpos = self.sim.qpos
    
    # Joint velocities
    qvel = self.sim.qvel

    # Goal
    goal = self.goal
    
    # Concatenate and return as a single observation vector
    observation = np.concatenate([qpos, qvel, goal])
    
    return observation

  def _get_reward(self):
    # reward function
    # euclidian distance between goal point and bracelet_with_vision_link which is the end of the arm
    cur_pos = self.sim.site_xpos[-1]
    # print(f"current_pos: {cur_pos}")

    dist = np.linalg.norm(self.goal - cur_pos)

    return -dist
