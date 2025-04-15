import numpy as np
import os
import mujoco
import mujoco.viewer
import gymnasium
from gymnasium.spaces import Box
import random
import time
import mink

class MujocoEnv(gymnasium.Env):
  def __init__(self, **kwargs):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kinova_gen3', 'scene.xml')
    self.model = mujoco.MjModel.from_xml_path(path)
    self.data = mujoco.MjData(self.model)

    # IK Config 
    self.configuration = mink.Configuration(self.model)

    # Move arm to reset position
    mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
    self.configuration.update(self.data.qpos)
    mujoco.mj_forward(self.model, self.data)
    # Initialize the mocap target at the end-effector site.
    mink.move_mocap_to_frame(self.model, self.data, "pinch_site_target", "pinch_site", "site")

    self.timestep = 0
    self.done = False
    self.cfg = kwargs['cfg']

    self.goal = self._generate_goal()

    self.max_episode_steps = self.cfg.max_episode_steps

    self.action_space = Box(-0.75, 0.75, (2,), np.float32)
    self.observation_space = Box(-np.inf, np.inf, (12,), np.float32)

    if self.cfg.viewer:
      self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
       
  def _generate_goal(self):
    """
    Generate goal, which is the desired position of the block
    """
    MIN_DISPLACEMENT = 0.1
    MAX_DISPLACEMENT = 0.5
    self.original_displacement = np.random.uniform(MIN_DISPLACEMENT, MAX_DISPLACEMENT)
    return self.data.body("cube_sweep").xpos.copy() + np.array([0, self.original_displacement, 0])

  def solve_ik(self, max_iters=30, pos_threshold=1e-4, ori_threshold=1e-4):
    """
    Solves the inverse kinematics to move the end-effector to the mocap body.

    Args:
        max_iters (int): Maximum number of iterations for the IK solver.
        pos_threshold (float): Position error threshold for convergence.
        ori_threshold (float): Orientation error threshold for convergence.
    """

    # Define the end-effector task
    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Set the target position and orientation
    T_wt = mink.SE3.from_mocap_name(self.model, self.data, "pinch_site_target")
    end_effector_task.set_target(T_wt)

    # Solve IK
    for i in range(max_iters):
        vel = mink.solve_ik(self.configuration, [end_effector_task], 0.01, "quadprog", 1e-2)
        self.configuration.integrate_inplace(vel, 0.01)

        # Check for convergence
        err = end_effector_task.compute_error(self.configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
        if pos_achieved and ori_achieved:
            break

    # Return whether the solver converged
    return self.configuration

  def step(self, action):
    action *= 0.01
    self.timestep += 1

    self.data.mocap_pos[0][:2] += action
    self.data.ctrl[:7] = self.solve_ik().q[:7]
    mujoco.mj_step(self.model, self.data)

    # Get the observation, reward, done, and info
    observation = self._get_observation()
    reward, success = self._get_reward()
    done = success
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
    self.timestep = 0
    self.done = False
    
    # Move arm to reset position
    mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
    self.configuration.update(self.data.qpos)
    mujoco.mj_forward(self.model, self.data)
    # Initialize the mocap target at the end-effector site.
    mink.move_mocap_to_frame(self.model, self.data, "pinch_site_target", "pinch_site", "site")

    # Generate goal 
    self.goal = self._generate_goal()
    
    obs = self._get_observation()
    reset_info = {}  # This can be populated with any reset-specific info if needed

    # update viewer
    if self.cfg.viewer: 
      self.viewer.sync()

    return obs, {}
        
  def _get_observation(self):
    # End-effector position
    pusher_pos = self.data.body('pusher').xpos.copy()

    # Goal
    goal = self.goal

    # Block position
    cube_pos = self.data.body('cube_sweep').xpos.copy()
    # Compute cube back position (-y direction from center)
    cube_half_length_y = self.model.geom('cube_geom_sweep').size[1]
    cube_xmat = self.data.body('cube_sweep').xmat.copy().reshape(3, 3)
    cube_back_pos = cube_pos - cube_half_length_y * cube_xmat[:, 1]

    # Block velocity (linear + angular)
    cube_vel = self.data.body('cube_sweep').cvel.copy()
    
    # Concatenate and return as a single observation vector
    observation = np.concatenate([goal - pusher_pos, cube_back_pos - pusher_pos, cube_vel])
    
    return observation

  def _get_reward(self):
    # reward function
    cube_pos = self.data.body('cube_sweep').xpos.copy()
    # Compute cube back position (-y direction from center)
    cube_half_length_y = self.model.geom('cube_geom_sweep').size[1]
    cube_xmat = self.data.body('cube_sweep').xmat.copy().reshape(3, 3)
    cube_back_pos = cube_pos - cube_half_length_y * cube_xmat[:, 1]
    pusher_pos = self.data.body('pusher').xpos.copy()
    # print(f"current_pos: {cur_pos}")
    cube_goal_dist = np.linalg.norm(self.goal - cube_pos)
    cube_arm_dist = np.linalg.norm(cube_back_pos - pusher_pos)

    rew = 2 - np.tanh(5*cube_goal_dist) - np.tanh(5*cube_arm_dist)
    if cube_goal_dist < self.cfg.goal_threshold and np.linalg.norm(self.data.body('cube_sweep').cvel.copy()) < self.cfg.vel_threshold:
      return rew, True
    else:
      return rew, False
