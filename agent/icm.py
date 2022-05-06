import torch
import torch.nn as nn
import utils

from agent.sac import SACAgent


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.backward_net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        # import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat, dim=-1, p=2, keepdim=True)
        backward_error = torch.norm(action - action_hat, dim=-1, p=2, keepdim=True)

        return forward_error, backward_error


class ICMAgent(SACAgent):
    def __init__(self, icm_scale, reward_free, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.reward_free = reward_free

        self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)

        # optimizers
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.icm.train()

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics["icm_loss"] = loss.item()
            metrics["icm_forward"] = forward_error.mean().item()
            metrics["icm_backward"] = backward_error.mean().item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        forward_error, _ = self.icm(obs, action, next_obs)

        reward = forward_error * self.icm_scale
        reward = torch.log(reward + 1.0)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device
        )

        metrics.update(self.update_icm(obs, action, next_obs, step))

        with torch.no_grad():
            intr_reward = self.compute_intr_reward(obs, action, next_obs, step)

        if self.use_tb or self.use_wandb:
            metrics["intr_reward"] = intr_reward.mean().item()

        if step % 1500 != 0:
            reward = intr_reward
        else:
            reward = extr_reward + intr_reward

        if self.use_tb or self.use_wandb:
            metrics["extr_reward"] = extr_reward.mean().item()
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor_and_alpha(obs, step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
