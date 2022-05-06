import torch
import torch.nn as nn
import utils

from agent.sac import SACAgent


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Linear_Encoder(nn.Module):
    def __init__(
        self, obs_dim, hidden_dim, output_dim, activation=nn.LeakyReLU(inplace=True)
    ):
        super().__init__()
        init_weights = 3e-3
        modules_state = []
        modules_state.append(nn.Linear(obs_dim, hidden_dim))
        modules_state.append(activation)

        # for i in range(hidden_layers - 1):
        # 	modules_state.append(nn.Linear(hidden_dim, hidden_dim))
        # 	modules_state.append(activation)

        last_fc = nn.Linear(hidden_dim, output_dim)
        last_fc.weight.data.uniform_(-init_weights, init_weights)
        last_fc.bias.data.uniform_(-init_weights, init_weights)
        modules_state.append(last_fc)

        self.fc = nn.Sequential(*modules_state)
        self.apply(weights_init_)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


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
    def __init__(self, icm_scale, sparsity, reward_free, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.reward_free = reward_free
        self.sparsity = sparsity

        if self.enc_type == "linear":
            self.l4_encoder = Linear_Encoder(
                self.obs_dim, self.hidden_dim, self.repr_dim
            ).to(self.device)
            self.obs_dim = self.repr_dim
        else:
            self.l4_encoder = nn.Identity()

        self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)

        # optimizers
        if self.enc_type == "linear":
            self.l4_encoder_opt = torch.optim.Adam(
                self.l4_encoder.parameters(), lr=self.lr
            )
        else:
            self.l4_encoder_opt = None

        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.icm.train()
        self.l4_encoder.train()

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        if self.l4_encoder_opt is not None:
            self.l4_encoder_opt.zero_grad(set_to_none=True)

        self.icm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_optimizer.step()

        if self.l4_encoder_opt is not None:
            self.l4_encoder_opt.step()

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

    def encode(self, obs):
        return self.l4_encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device
        )

        state = self.encode(obs)
        with torch.no_grad():
            next_state = self.encode(next_obs)

        metrics.update(self.update_icm(state, action, next_state, step))

        with torch.no_grad():
            intr_reward = self.compute_intr_reward(state, action, next_state, step)

        if self.use_tb or self.use_wandb:
            metrics["intr_reward"] = intr_reward.mean().item()

        if step % self.sparsity != 0:
            reward = intr_reward
        else:
            reward = extr_reward + intr_reward

        if self.use_tb or self.use_wandb:
            metrics["extr_reward"] = extr_reward.mean().item()
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(state.detach(), action, reward, discount, next_state.detach(), step)
        )

        # update actor
        metrics.update(self.update_actor_and_alpha(state.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
