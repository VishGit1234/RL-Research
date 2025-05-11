import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import os, shutil, pickle
from kinova_env import KinovaEnv


def train():
    MAX_ITERATIONS=101
    NUM_ENVS=512
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

    gs.init(backend=gs.gpu, logging_level="info")

    log_dir = "logs/kinova_sweep"

    env_cfg = {
        "episode_length_s" : 10,
        "init_joint_angles" : np.array([6.25032076, 0.60241784, 3.15709118, -2.128586102, 6.28220792, -0.39995964788322566, 1.55241801, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822]),
        "init_quat" : np.array([1, 0, 0., 1]),
        "bracelet_link_height=0.25" : 0.25,
        "init_box_pos" : (0.2, 0.2, 0.02),
        "box_size" : (0.08, 0.08, 0.02),
        "clip_actions" : 0.01,
        "termination_if_cube_goal_dist_less_than" : 0.001,
        "cube_goal_dist_rew_scale" : 3,
        "cube_arm_dist_rew_scale" : 2,
        "success_reward" : 100,
        "target_displacement" : 0.1
    }    

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = KinovaEnv(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        show_viewer=False
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=MAX_ITERATIONS, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()