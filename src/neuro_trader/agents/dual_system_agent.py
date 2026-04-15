"""Dual-system trading agent with dynamic emotional/rational weighting.

System-1 (emotional): fast, volatility-sensitive responses.
System-2 (rational): trend-following, stability-seeking responses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class DualSystemConfig:
    """Configuration for the dual-system policy combiner."""

    state_dim: int
    action_dim: int = 3
    hidden_dim: int = 128
    emotional_vol_sensitivity: float = 2.5
    rational_trend_sensitivity: float = 2.0
    min_system_weight: float = 0.1


class _SubPolicy(nn.Module):
    """Shared sub-policy architecture used by both systems."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DualSystemAgent(nn.Module):
    """Combines emotional and rational systems with dynamic weighting.

    State assumptions (from TradingEnv observation):
      idx 0: returns
      idx 1: fear
      idx 2: greed
      idx 3: stress (volatility proxy)
    """

    def __init__(self, config: DualSystemConfig) -> None:
        super().__init__()
        self.config = config
        self.emotional_system = _SubPolicy(
            input_dim=config.state_dim,
            output_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )
        self.rational_system = _SubPolicy(
            input_dim=config.state_dim,
            output_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(self, state: Tensor) -> Tensor:
        """Return blended action values with dynamic system weighting."""
        emotional_q = self.emotional_system(state)
        rational_q = self.rational_system(state)

        emotional_w, rational_w = self.compute_system_weights(state)
        blended_q = emotional_w * emotional_q + rational_w * rational_q
        return blended_q

    def compute_system_weights(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Compute dynamic system weights from volatility and trend stability.

        Rules:
        - Emotional system dominates when volatility (stress) is high.
        - Rational system dominates when trend is stable.
        - Weights are dynamic and sum to 1.
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)

        stress = state[:, 3].clamp(min=0.0)  # volatility proxy
        returns = state[:, 0]

        # Stable trend proxy: lower stress + consistent return direction.
        trend_consistency = returns.abs()
        trend_stability = (1.0 - stress).clamp(min=0.0) * trend_consistency

        emotional_score = torch.sigmoid(self.config.emotional_vol_sensitivity * stress)
        rational_score = torch.sigmoid(self.config.rational_trend_sensitivity * trend_stability)

        scores = torch.stack([emotional_score, rational_score], dim=1)
        weights = torch.softmax(scores, dim=1)

        emotional_w = weights[:, 0:1]
        rational_w = weights[:, 1:2]

        # Keep both systems active with a minimum blend floor.
        min_w = self.config.min_system_weight
        emotional_w = (1.0 - 2.0 * min_w) * emotional_w + min_w
        rational_w = (1.0 - 2.0 * min_w) * rational_w + min_w

        # Re-normalize after floor adjustment.
        norm = emotional_w + rational_w
        emotional_w = emotional_w / norm
        rational_w = rational_w / norm

        return emotional_w, rational_w

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        """Select greedy action from blended system values."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.forward(state_t)
        return int(torch.argmax(q_values, dim=1).item())
