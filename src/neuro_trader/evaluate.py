"""Evaluation script for DQN trading agent vs random baseline.

Workflow:
1) Download data and compute behavioral features
2) Split into train/test segments (test is unseen)
3) Train DQN on train segment
4) Evaluate DQN and random strategy on test segment
5) Print metrics and save comparison plots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agents import DQNConfig
from .behavioral_signals import compute_behavioral_signals
from .data_loader import download_price_data
from .evaluation_metrics import compute_all_metrics
from .trading_env import TradingEnv
from .train_dqn import train_dqn_agent


@dataclass
class EvaluationResult:
    """Container for one policy's evaluation output."""

    name: str
    metrics: dict[str, float]
    portfolio_values: list[float]
    actions: list[int]




def run_baseline_buy_hold(data: pd.DataFrame) -> pd.Series:
    """Simple buy-and-hold baseline cumulative return series."""
    return data["Close"].pct_change().cumsum()


def _split_train_test(data: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be within [0.1, 0.95].")
    split_idx = int(len(data) * train_ratio)
    # Equivalent split intent: train_data = data[:70%], test_data = data[70%:]
    if split_idx < 2 or (len(data) - split_idx) < 2:
        raise ValueError("Not enough rows to form train/test sets with at least 2 rows each.")
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    return train_df, test_df


def _evaluate_dqn_on_env(agent, env: TradingEnv) -> EvaluationResult:
    state, info = env.reset(seed=123)
    done = False
    actions: list[int] = []
    portfolio_values: list[float] = [float(info["portfolio_value"])]

    while not done:
        action = agent.select_action(state, explore=False)
        actions.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(float(info["portfolio_value"]))
        done = terminated or truncated

    metrics = compute_all_metrics(portfolio_values)
    return EvaluationResult(name="DQN", metrics=metrics, portfolio_values=portfolio_values, actions=actions)


def _evaluate_random_on_env(env: TradingEnv, seed: int = 123) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    state, info = env.reset(seed=seed)
    done = False
    actions: list[int] = []
    portfolio_values: list[float] = [float(info["portfolio_value"])]

    while not done:
        action = int(rng.integers(0, env.action_space.n))
        actions.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(float(info["portfolio_value"]))
        done = terminated or truncated

    metrics = compute_all_metrics(portfolio_values)
    return EvaluationResult(name="Random", metrics=metrics, portfolio_values=portfolio_values, actions=actions)


def _save_plots(results: list[EvaluationResult], test_data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: portfolio value trajectories.
    plt.figure(figsize=(10, 5))
    for result in results:
        plt.plot(result.portfolio_values, label=result.name)
    plt.title("Portfolio Value on Unseen Test Data")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_value_over_time.png", dpi=160)
    plt.close()

    # Plot 2: stock price with buy/sell markers for DQN policy.
    dqn_result = next(result for result in results if result.name == "DQN")
    price_series = test_data["Close"].reset_index(drop=True)
    action_idx = np.arange(len(dqn_result.actions))
    buy_idx = action_idx[np.array(dqn_result.actions) == 2]
    sell_idx = action_idx[np.array(dqn_result.actions) == 0]

    plt.figure(figsize=(12, 5))
    plt.plot(price_series.values, label="Close Price", color="steelblue")
    if len(buy_idx) > 0:
        plt.scatter(buy_idx, price_series.iloc[buy_idx], marker="^", color="green", s=45, label="Buy")
    if len(sell_idx) > 0:
        plt.scatter(sell_idx, price_series.iloc[sell_idx], marker="v", color="red", s=45, label="Sell")
    plt.title("Test Stock Price with DQN Buy/Sell Markers")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stock_price_with_trades.png", dpi=160)
    plt.close()

    # Plot 3: behavioral signals over time on test data.
    plt.figure(figsize=(12, 5))
    plt.plot(test_data["fear"].reset_index(drop=True), label="Fear")
    plt.plot(test_data["greed"].reset_index(drop=True), label="Greed")
    plt.plot(test_data["stress"].reset_index(drop=True), label="Stress")
    plt.title("Behavioral Signals Over Time (Test Set)")
    plt.xlabel("Step")
    plt.ylabel("Signal Value (0-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "behavioral_signals_over_time.png", dpi=160)
    plt.close()


def run_evaluation(
    ticker: str = "AAPL",
    start: str = "2018-01-01",
    lookback: int = 20,
    train_ratio: float = 0.7,
    episodes: int = 30,
    output_dir: str = "artifacts",
) -> dict[str, EvaluationResult]:
    """Train on train split, evaluate on unseen split, compare against random policy."""
    prices = download_price_data(ticker=ticker, start=start)
    train_prices, test_prices = _split_train_test(prices, train_ratio=train_ratio)

    train_data, normalization_bounds = compute_behavioral_signals(
        train_prices,
        lookback=lookback,
        return_normalization_bounds=True,
    )
    test_data = compute_behavioral_signals(
        test_prices,
        lookback=lookback,
        normalization_bounds=normalization_bounds,
    )

    baseline_buy_hold_returns = run_baseline_buy_hold(test_data).dropna()

    train_env = TradingEnv(train_data, initial_capital=10_000.0, transaction_cost_pct=0.001)

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
    agent, training_metrics = train_dqn_agent(train_env, episodes=episodes, config=config)

    test_env_for_dqn = TradingEnv(test_data, initial_capital=10_000.0, transaction_cost_pct=0.001)
    test_env_for_random = TradingEnv(test_data, initial_capital=10_000.0, transaction_cost_pct=0.001)

    dqn_result = _evaluate_dqn_on_env(agent, test_env_for_dqn)
    random_result = _evaluate_random_on_env(test_env_for_random)

    print(f"Training completed on {len(train_data)} rows; evaluation on unseen {len(test_data)} rows.")
    print("Train mean reward:", float(np.mean(training_metrics.episode_rewards)))
    if not baseline_buy_hold_returns.empty:
        print(
            "Buy/Hold baseline cumulative return:",
            float(baseline_buy_hold_returns.iloc[-1]),
        )

    for result in [dqn_result, random_result]:
        print(f"\n{result.name} metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value:.6f}")

    _save_plots([dqn_result, random_result], test_data=test_data, output_dir=Path(output_dir))
    print(f"\nSaved plots to: {Path(output_dir).resolve()}")

    return {"dqn": dqn_result, "random": random_result}


if __name__ == "__main__":
    run_evaluation()
