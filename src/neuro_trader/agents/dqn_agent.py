"""DQN agent components for discrete-action trading.

Designed to stay modular for future dual-system extensions:
- network separated from agent logic
- replay buffer separated from optimization logic
- config dataclass for easy experimentation
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DQNConfig:
    """Hyperparameter container for DQN training."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    target_update_tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 128


class QNetwork(nn.Module):
    """Simple MLP Q-network mapping state -> Q-values for buy/hold/sell."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ReplayBuffer:
    """Uniform replay buffer for off-policy DQN learning."""

    def __init__(self, capacity: int) -> None:
        self._storage: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._storage, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class DQNAgent:
    """DQN agent for buy/hold/sell action selection and optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim=self.config.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim=self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=self.config.buffer_size)

        self.epsilon = self.config.epsilon_start
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action with epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def optimize(self) -> float | None:
        """Run one gradient update; returns loss or None if buffer too small."""
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update_target_network()
        self._decay_epsilon()
        return float(loss.item())

    def _soft_update_target_network(self) -> None:
        tau = self.config.target_update_tau
        for target_param, source_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
