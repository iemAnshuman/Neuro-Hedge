import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class OUNoise:
    """Ornstein-Uhlenbeck process for better exploration"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, np.array([done]))
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """FIXED: Deeper network with batch normalization"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.layer_2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.layer_3 = nn.Linear(300, action_dim)
        
        # Initialize weights
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.bn1(self.layer_1(state)))
        x = F.relu(self.bn2(self.layer_2(x)))
        action = torch.sigmoid(self.layer_3(x))
        return action

class Critic(nn.Module):
    """FIXED: Deeper network with batch normalization"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State pathway
        self.layer_1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        
        # Combined pathway (state + action)
        self.layer_2 = nn.Linear(400 + action_dim, 300)
        self.layer_3 = nn.Linear(300, 1)
        
        # Initialize weights
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)
        self.layer_3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.layer_1(state)))
        x = torch.cat([x, action], 1)
        x = F.relu(self.layer_2(x))
        value = self.layer_3(x)
        return value

class DDPGAgent:
    def __init__(self, state_dim, action_dim, tau=0.001, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3):
        
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2)

        self.tau = tau
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # FIXED: Add Ornstein-Uhlenbeck noise for better exploration
        self.noise = OUNoise(action_dim)

    def select_action(self, state, add_noise=False):
        self.actor.eval()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t = self.actor(state_t)
        action = action_t.cpu().numpy().flatten()
        
        if add_noise:
            action = action + self.noise.sample()
            
        return np.clip(action, 0.0, 1.0)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth", map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth", map_location=torch.device('cpu')))
        self.actor.eval()
        self.critic.eval()

    def update_parameters(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.batch_size:
            return None, None

        self.actor.train()
        self.critic.train()

        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q_values * (1 - dones))
        
        current_q = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")