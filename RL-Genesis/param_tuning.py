import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import os, shutil, pickle
from kinova_env import KinovaEnv
from config import env_cfg, train_cfg, MAX_ITERATIONS
import torch
import optuna
from functools import partial

def objective(trial, env):
    # PPO algorithm hyperparameters
    train_cfg["algorithm"]["class_name"] = "PPO"
    train_cfg["policy"]["class_name"] = "ActorCritic"
    train_cfg["algorithm"]["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    train_cfg["algorithm"]["gamma"] = trial.suggest_uniform("gamma", 0.90, 0.999)
    train_cfg["algorithm"]["lam"] = trial.suggest_uniform("lam", 0.90, 0.99)
    train_cfg["algorithm"]["entropy_coef"] = trial.suggest_loguniform("entropy_coef", 1e-5, 1e-2)
    train_cfg["algorithm"]["clip_param"] = trial.suggest_uniform("clip_param", 0.05, 0.3)
    train_cfg["algorithm"]["value_loss_coef"] = trial.suggest_uniform("value_loss_coef", 0.1, 2.0)
    train_cfg["algorithm"]["num_learning_epochs"] = trial.suggest_int("num_learning_epochs", 1, 10)

    # Policy network hyperparameters
    actor_dim = trial.suggest_categorical("actor_hidden_dim", [64, 128, 256])
    critic_dim = trial.suggest_categorical("critic_hidden_dim", [64, 128, 256])
    train_cfg["policy"]["actor_hidden_dims"] = [actor_dim, actor_dim]
    train_cfg["policy"]["critic_hidden_dims"] = [critic_dim, critic_dim]

    # Run the training with the current hyperparameters
    try:
        reward = train(env)
    except:
        print("Training failed for trial:", trial.number)
        return -1

    return reward

def train(env):
    log_dir = "logs/kinova_sweep"  

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env.reset()
    print(train_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=MAX_ITERATIONS, init_at_random_ep_len=True)

    policy = runner.get_inference_policy()

    obs, _ = env.reset()
    episode_reward = 0
    for i in range(int(env_cfg["episode_length_s"] / env.dt)):
        action = policy(obs)
        obs, reward, done, info = env.step(actions=action)
        episode_reward += torch.mean(reward).item()
    
    return episode_reward

def main():
    NUM_ENVS=2048
    gs.init(backend=gs.gpu, logging_level="warning")

    env = KinovaEnv(
        num_envs=NUM_ENVS,
        env_cfg=env_cfg,
        show_viewer=False
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, env=env), n_trials=100)

    best_trial = study.best_trial

    objective(best_trial, env)

if __name__ == "__main__":
    main()