import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import os, pickle
from kinova_env import KinovaEnv
from config import env_cfg, train_cfg, MAX_ITERATIONS

def eval():
    NUM_ENVS=3

    gs.init(backend=gs.gpu, logging_level="info")

    log_dir = "logs/kinova_sweep"

    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} does not exist. Please run the training script first.")

    env = KinovaEnv(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        show_viewer=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, "model_100.pt"))

    policy = runner.get_inference_policy()

    obs, _ = env.reset()
    for i in range(10000):
        action = policy(obs)
        obs, reward, done, info = env.step(actions=action)

if __name__ == "__main__":
    eval()