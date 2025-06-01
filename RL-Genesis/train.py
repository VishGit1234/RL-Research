import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import os, shutil, pickle
from kinova_env import KinovaEnv
from kinova_env_opt import KinovaEnvOpt
from config import env_cfg, train_cfg, MAX_ITERATIONS, Struct


def train():
    NUM_ENVS=4096

    gs.init(backend=gs.gpu, logging_level="warning")

    log_dir = "logs/kinova_sweep"  

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    env = KinovaEnvOpt(
        num_envs=NUM_ENVS,
        env_cfg=Struct(**env_cfg),
        show_viewer=False
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=MAX_ITERATIONS, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()