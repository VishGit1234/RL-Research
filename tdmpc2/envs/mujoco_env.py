import numpy as np
import os
import mujoco
import mujoco.viewer
import gymnasium
from gymnasium.spaces import Box
import random
from copy import deepcopy

class MujocoEnv(gymnasium.Env):
  def __init__(self, **kwargs):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kinova_gen3', 'scene.xml')
    self.model = mujoco.MjModel.from_xml_path(path)
    self.sim = mujoco.MjData(self.model)
    # goal should be the position we want the end of the arm to be at
    # should be an numpy array
    self.timestep = 0
    self.done = False
    self.cfg = kwargs['cfg']

    self.goal = self._generate_goal()

    self.max_episode_steps = self.cfg.max_episode_steps

    self.action_space = Box(-0.75, 0.75, (3,), np.float32)
    self.observation_space = Box(-np.inf, np.inf, (17,), np.float32)

    if self.cfg.viewer:
      self.viewer = mujoco.viewer.launch_passive(self.model, self.sim)
      i = self.viewer.user_scn.ngeom
      mujoco.mjv_initGeom(
        self.viewer.user_scn.geoms[i],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0, 0],
        pos=self.goal,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 2])
      )
      self._geom_id = i
      self.viewer.user_scn.ngeom = i + 1
       
  def _generate_goal(self):
    MIN_RADIUS = 0.4
    MAX_RADIUS = 0.6

    return np.array([0.3, 0, 0.8])

    return np.array([(random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                    (random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                    random.uniform(MIN_RADIUS, MAX_RADIUS)])

  def step(self, action):
    # print(self.goal)
    self.timestep += 1

    # Apply the action to the environment
    # self.sim.ctrl[:] = action # np.zeros(action.shape)
    # cur_pos = self.sim.site('pinch_site').xpos.copy()
    # action = 0.01*(self.goal - cur_pos) / np.linalg.norm(self.goal - cur_pos)
    self.sim.mocap_pos[0] = self.goal
    self.sim.mocap_quat[0] = np.array([0, 1, 0, 0])
    print(self.goal, self.sim.mocap_pos, self.sim.body('bracelet_link').xpos)
    mujoco.mj_step(self.model, self.sim)

    # Get the observation, reward, done, and info
    observation = self._get_observation()
    reward, success = self._get_reward(action)
    done = False
    self.done = done
    truncated = False
    info = {}
    info['success'] = success

    # update viewer
    if self.cfg.viewer: self.viewer.sync()

    if self.timestep > self.max_episode_steps:
      done = True
      self.done = True

    return observation, reward, done, truncated, info

  def reset(self, **kwargs):
    # Reset MuJoCo
    mujoco.mj_resetData(self.model, self.sim)
    mujoco.mj_forward(self.model, self.sim)

    self.timestep = 0
    self.done = False

    # Get observation 
    self.goal = self._generate_goal()
    if self.cfg.viewer: self.viewer.user_scn.geoms[self._geom_id].pos[:] = self.goal
    
    obs = self._get_observation()
    reset_info = {}  # This can be populated with any reset-specific info if needed

    # update viewer
    if self.cfg.viewer: 
      self.viewer.sync()

    return obs, {}
        
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

  def _get_reward(self, action):
    # reward function
    # euclidian distance between goal point and bracelet_with_vision_link which is the end of the arm
    cur_pos = self.sim.site_xpos[0]
    # print(f"current_pos: {cur_pos}")
    dist = np.linalg.norm(self.goal - cur_pos)

    # # minimize actuator movement (mean squared average of actuator movement)
    # msa = np.mean(np.square(action))

    # reward = np.clip(-dist - msa, -1000, 1000)
    rew = 1 - np.tanh(5*dist)
    if dist < self.cfg.goal_threshold and np.mean(np.abs(self.sim.qvel)) < self.cfg.vel_threshold:
      return 10 + rew, True
    else:
      return rew, False
