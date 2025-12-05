import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # simple replay buffer (you can move this to a separate file if you want)
        self.buffer = deque(maxlen=buffer_size)

    def epsilon(self) -> float:
        # exponential decay
        frac = self.steps_done / self.epsilon_decay
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-frac)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        self.steps_done += 1

        if (not greedy) and (random.random() < self.epsilon()):
            return random.randrange(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def can_update(self) -> bool:
        return len(self.buffer) >= self.batch_size

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def update(self, global_step: int):
        if not self.can_update():
            return None

        states, actions, rewards, next_states, dones = self.sample_batch()

        # current Q estimates
        q_values = self.online_net(states).gather(1, actions)

        # target Q using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # update target network
        if global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())
