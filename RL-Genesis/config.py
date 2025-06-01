import numpy as np

MAX_ITERATIONS = 100
train_cfg = {
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.1,
        "desired_kl": 0.01,
        "entropy_coef": 0.001,
        "gamma": 0.95,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 0.5,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "init_member_classes": {},
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [128, 128],
        "critic_hidden_dims": [128, 128],
        "init_noise_std": 2.0,
        "class_name": "ActorCritic",
        "noise_std_type": "log"
    },
    "runner": {
        "checkpoint": -1,
        "experiment_name": "kinova_sweep",
        "load_run": -1,
        "log_interval": 1,
        "max_iterations": MAX_ITERATIONS,
        "record_interval": -1,
        "resume": False,
        "resume_path": None,
        "run_name": "",
    },
    "runner_class_name": "OnPolicyRunner",
    "num_steps_per_env": 24,
    "save_interval": 50,
    "empirical_normalization": None,
    "seed": 29,
}

env_cfg = {
    "episode_length_s" : 2,
    "init_joint_angles" : np.array([6.9761, 1.1129, 1.7474, -2.2817, 7.5884, -1.1489, 1.6530, 0.8213, 0.8200, 0.8209, 0.8208, 0.8217, 0.8210]),
    "init_quat" : np.array([0, 0, 0., 1]),
    "bracelet_link_height" : 0.25,
    "init_box_pos" : (0.2, 0.2, 0.02),
    "box_size" : (0.08, 0.08, 0.02),
    "clip_actions" : 0.01,
    "termination_if_cube_goal_dist_less_than" : 0.01,
    "cube_goal_dist_rew_scale" : 10,
    "cube_arm_dist_rew_scale" : 10,
    "success_reward" : 10,
    "target_displacement" : 0.3,
    "action_scale" : 0.01,
    "robot_mjcf_file" : ".\kinova_gen3\gen3.xml",
}  

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

import torch
def check_for_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError("Tensor contains NaN values")
    return tensor