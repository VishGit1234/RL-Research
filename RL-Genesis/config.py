import numpy as np

MAX_ITERATIONS = 200
train_cfg = {
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "init_member_classes": {},
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
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
    "save_interval": 100,
    "empirical_normalization": None,
    "seed": 1,
}

env_cfg = {
    "episode_length_s" : 10,
    "init_joint_angles" : np.array([6.25032076, 0.60241784, 3.15709118, -2.128586102, 6.28220792, -0.39995964788322566, 1.55241801, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822]),
    "init_quat" : np.array([1, 0, 0., 1]),
    "bracelet_link_height" : 0.25,
    "init_box_pos" : (0.2, 0.2, 0.02),
    "box_size" : (0.08, 0.08, 0.02),
    "clip_actions" : 0.005,
    "termination_if_cube_goal_dist_less_than" : 0.01,
    "cube_goal_dist_rew_scale" : 5,
    "cube_arm_dist_rew_scale" : 1,
    "success_reward" : 1000,
    "target_displacement" : 0.1
}  

import torch
def check_for_nan(tensor):
    if torch.isnan(tensor).any():
        raise ValueError("Tensor contains NaN values")
    return tensor