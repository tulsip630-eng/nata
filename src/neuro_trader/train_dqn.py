"""Training utilities for DQN-based neuro-adaptive trading agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import DQNAgent, DQNConfig
from .behavioral_signals import compute_behavioral_signals
from .data_loader import download_price_data
from .evaluation_metrics import compute_all_metrics
from .trading_env import TradingEnv


@dataclass
class TrainingMetrics:
    """Container for high-level training diagnostics."""

    episode_rewards: list[float]
    episode_final_values: list[float]
    losses: list[float]
    episode_cumulative_returns: list[float]
    episode_sharpes: list[float]
    episode_max_drawdowns: list[float]
    episode_volatilities: list[float]


def train_dqn_agent(
    env: TradingEnv,
    episodes: int = 20,
    config: DQNConfig | None = None,
) -> tuple[DQNAgent, TrainingMetrics]:
    """Train a DQN agent on a provided trading environment."""
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    episode_rewards: list[float] = []
    episode_final_values: list[float] = []
    losses: list[float] = []
    episode_cumulative_returns: list[float] = []
    episode_sharpes: list[float] = []
    episode_max_drawdowns: list[float] = []
    episode_volatilities: list[float] = []

    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        episode_losses: list[float] = []

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        if episode_losses:
            losses.append(float(np.mean(episode_losses)))

        portfolio_values = [float(x) for x in info["portfolio_value_history"]]
        episode_final_values.append(portfolio_values[-1])

        episode_metric = compute_all_metrics(portfolio_values)
        episode_cumulative_returns.append(float(episode_metric["cumulative_return"]))
        episode_sharpes.append(float(episode_metric["sharpe_ratio"]))
        episode_max_drawdowns.append(float(episode_metric["max_drawdown"]))
        episode_volatilities.append(float(episode_metric["volatility"]))

    metrics = TrainingMetrics(
        episode_rewards=episode_rewards,
        episode_final_values=episode_final_values,
        losses=losses,
        episode_cumulative_returns=episode_cumulative_returns,
        episode_sharpes=episode_sharpes,
        episode_max_drawdowns=episode_max_drawdowns,
        episode_volatilities=episode_volatilities,
    )
    return agent, metrics


def run_dqn_pipeline(
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    lookback: int = 20,
    episodes: int = 20,
) -> tuple[DQNAgent, TrainingMetrics]:
    """Build data/features/env stack and train DQN end-to-end."""
    data = download_price_data(ticker=ticker, start=start)
    features = compute_behavioral_signals(data, lookback=lookback)

    env = TradingEnv(
        features,
        initial_capital=10_000.0,
        max_position=5,
        transaction_cost_pct=0.001,
    )

    config = DQNConfig(
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=10_000,
        min_buffer_size=256,
        target_update_tau=0.01,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        hidden_dim=128,
    )

    agent, metrics = train_dqn_agent(env=env, episodes=episodes, config=config)

    print(f"Training finished for {episodes} episodes.")
    print(f"Mean episode reward: {float(np.mean(metrics.episode_rewards)):.4f}")
    print(f"Mean final portfolio value: {float(np.mean(metrics.episode_final_values)):.2f}")
    print(f"Mean cumulative return: {float(np.mean(metrics.episode_cumulative_returns)):.4f}")
    print(f"Mean Sharpe ratio: {float(np.mean(metrics.episode_sharpes)):.4f}")
    print(f"Mean max drawdown: {float(np.mean(metrics.episode_max_drawdowns)):.4f}")
    print(f"Mean volatility: {float(np.mean(metrics.episode_volatilities)):.4f}")
    if metrics.losses:
        print(f"Final loss: {metrics.losses[-1]:.6f}")

    return agent, metrics


if __name__ == "__main__":
    run_dqn_pipeline()
