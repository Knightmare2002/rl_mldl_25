import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc_critic = nn.Sequential(
                            nn.Linear(state_space, self.hidden),
                            nn.Tanh(),
                            nn.Linear(self.hidden, 1)
                        )


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        value = self.fc_critic(x)

        
        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.lr = 1e-3
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.device = device

    def update_policy(self,previous_state, state, action_probabilities, reward, done):
        state = torch.from_numpy(state).float().to(self.device)
        previous_state = torch.from_numpy(previous_state).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        
        
        _, next_state_value = self.policy(state) 
        _, current_state_value = self.policy(previous_state)

        next_state_value = next_state_value.squeeze()
        current_state_value = current_state_value.squeeze()

        target = reward + self.gamma*next_state_value*(1-int(done))
        adv = target - current_state_value
        loss_crit = F.mse_loss(current_state_value,target.detach())
        loss_actor = -(adv.detach() * action_probabilities) 

        loss = loss_crit + loss_actor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob



