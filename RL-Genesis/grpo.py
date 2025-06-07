import torch
import torch.nn as nn
from torch.distributions import Normal
from kinova_env_opt import KinovaEnvOpt
from config import env_cfg, Struct
import genesis as gs

class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim*2)  # Output mean and log_std for actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    def act(self, obs):
        out = self(obs)
        mean, log_std = out.chunk(2, dim=-1)
        dist = Normal(mean, torch.exp(log_std))
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class GRPO():
    def __init__(self, env: KinovaEnvOpt, obs_dim, action_dim):
        self.env = env
        self.num_envs = env.num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy: Policy = Policy(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)

    def rollout(self):
        obs, _ = self.env.reset()
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_log_probs = []
        while not done.any():
            action, log_prob = self.act(obs)
            obs, reward, done, _ = self.env.step(action)
            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
        assert done.all(), "All environments must terminate at same time."
        return (
            torch.stack(ep_obs, dim=1),
            torch.stack(ep_actions, dim=1), 
            torch.sum(torch.stack(ep_rewards, dim=1), dim=1),
            torch.stack(ep_log_probs, dim=1)
        )

    def act(self, obs):
        return self.policy.act(obs)

    def compute_loss(self, obs, old_log_probs, advantages):
        new_actions, new_log_probs = self.act(obs.view(-1, self.obs_dim))
        ratio = (new_log_probs.view(self.num_envs, -1, self.action_dim)- old_log_probs).exp()
        surrogate_loss = ratio * advantages.view(self.num_envs, 1, 1)
        clipped_surrogate_loss = torch.clamp(ratio, 0.8, 1.2) * advantages.view(self.num_envs, 1, 1)
        loss = torch.min(surrogate_loss, clipped_surrogate_loss).mean() - 0.001 * new_log_probs.mean()  # Entropy regularization
        return -loss

    def compute_advantages(self, rewards):
        return (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    def train(self, num_epochs, num_iterations=5):
        for epoch in range(num_epochs):
            # Perform rollout (batch size = num envs)
            with torch.no_grad():
                obs, actions, rewards, log_probs = self.rollout()
                # Compute advantages
                advantages = self.compute_advantages(rewards)
            for i in range(num_iterations):
                # Update policy
                loss = self.compute_loss(obs, log_probs, advantages)
                self.policy.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Reward: {rewards.mean().item():.4f}")

if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level="warn")
    env_cfg["episode_length_s"] = 1
    env = KinovaEnvOpt(num_envs=1024, env_cfg=Struct(**env_cfg), show_viewer=True)
    obs_dim = env.num_obs
    action_dim = env.num_actions

    grpo = GRPO(env, obs_dim, action_dim)
    grpo.train(num_epochs=100, num_iterations=10)