"""Gymnasium trading environment with behavioral-state observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class PortfolioState:
    """Container for mutable portfolio values."""

    cash: float
    position: int
    portfolio_value: float


class TradingEnv(gym.Env[np.ndarray, int]):
    """Simple buy/hold/sell environment for a single asset.

    Action space:
        0 -> sell one unit
        1 -> hold
        2 -> buy one unit

    Observation vector:
        [return, fear, greed, stress, normalized_position, normalized_cash,
         is_short, is_cash, is_long]

    Reward:
        Change in portfolio value from previous step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10_000.0,
        max_position: int = 10,
        close_col: str = "Close",
        transaction_cost_pct: float = 0.001,
    ) -> None:
        required = {close_col, "returns", "fear", "greed", "stress"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Input data missing required feature columns: {sorted(missing)}")

        if len(data) < 2:
            raise ValueError("Trading environment requires at least two rows of data.")

        self.data = data.reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.max_position = int(max_position)
        self.close_col = close_col
        self.transaction_cost_pct = float(transaction_cost_pct)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_idx = 0
        self._portfolio = PortfolioState(
            cash=self.initial_capital,
            position=0,
            portfolio_value=self.initial_capital,
        )
        self._portfolio_value_history: list[float] = []
        self._cost_history: list[float] = []

    @property
    def portfolio_value_history(self) -> list[float]:
        """Portfolio value trajectory for the active episode."""
        return self._portfolio_value_history

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment state and return initial observation."""
        super().reset(seed=seed)
        self._step_idx = 0
        self._portfolio = PortfolioState(
            cash=self.initial_capital,
            position=0,
            portfolio_value=self.initial_capital,
        )
        self._portfolio_value_history = [self.initial_capital]
        self._cost_history = [0.0]
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action, advance one timestep, and compute reward."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

        trade_price = float(self.data.loc[self._step_idx, self.close_col])
        prev_portfolio_value = self._portfolio.portfolio_value

        transaction_cost = 0.0
        can_buy = self._portfolio.position < self.max_position
        can_sell = self._portfolio.position > -self.max_position

        if action == 2 and can_buy:
            gross_cash_change = -trade_price
            transaction_cost = abs(trade_price) * self.transaction_cost_pct
            if self._portfolio.cash + gross_cash_change - transaction_cost >= 0.0:
                self._portfolio.position += 1
                self._portfolio.cash += gross_cash_change - transaction_cost
            else:
                transaction_cost = 0.0

        elif action == 0 and can_sell:
            gross_cash_change = trade_price
            transaction_cost = abs(trade_price) * self.transaction_cost_pct
            self._portfolio.position -= 1
            self._portfolio.cash += gross_cash_change - transaction_cost

        self._step_idx += 1
        terminated = self._step_idx >= len(self.data) - 1

        mark_price = float(self.data.loc[self._step_idx, self.close_col])
        self._portfolio.portfolio_value = self._portfolio.cash + self._portfolio.position * mark_price

        # Reward is change in portfolio value.
        reward = self._portfolio.portfolio_value - prev_portfolio_value

        self._portfolio_value_history.append(self._portfolio.portfolio_value)
        self._cost_history.append(transaction_cost)

        observation = self._get_observation()
        info = self._get_info()
        truncated = False

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        row = self.data.loc[self._step_idx]
        normalized_position = self._portfolio.position / max(self.max_position, 1)
        normalized_cash = self._portfolio.cash / max(self.initial_capital, 1e-8)
        is_short, is_cash, is_long = self._position_state_flags(self._portfolio.position)

        return np.array(
            [
                float(row["returns"]),
                float(row["fear"]),
                float(row["greed"]),
                float(row["stress"]),
                float(normalized_position),
                float(normalized_cash),
                float(is_short),
                float(is_cash),
                float(is_long),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _position_state_flags(position: int) -> tuple[int, int, int]:
        """Map signed position to [short, cash, long] indicator flags."""
        if position < 0:
            return 1, 0, 0
        if position > 0:
            return 0, 0, 1
        return 0, 1, 0

    def _get_info(self) -> dict[str, Any]:
        is_short, is_cash, is_long = self._position_state_flags(self._portfolio.position)
        return {
            "step": self._step_idx,
            "cash": self._portfolio.cash,
            "position": self._portfolio.position,
            "position_state": "short" if is_short else "long" if is_long else "cash",
            "portfolio_value": self._portfolio.portfolio_value,
            "portfolio_value_history": self._portfolio_value_history,
            "latest_transaction_cost": self._cost_history[-1],
            "cumulative_transaction_cost": float(sum(self._cost_history)),
        }

    def render(self) -> None:
        """Print the current portfolio state."""
        info = self._get_info()
        print(
            f"step={info['step']} cash={info['cash']:.2f} position={info['position']} "
            f"state={info['position_state']} portfolio_value={info['portfolio_value']:.2f} "
            f"cum_cost={info['cumulative_transaction_cost']:.4f}"
        )
