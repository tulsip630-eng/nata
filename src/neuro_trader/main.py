"""Run a minimal end-to-end neuro-adaptive trading pipeline."""

from __future__ import annotations

from .behavioral_signals import compute_behavioral_signals
from .data_loader import download_price_data
from .evaluation_metrics import compute_all_metrics
from .trading_env import TradingEnv


def _heuristic_action(fear: float, greed: float) -> int:
    """Tiny baseline policy: buy on greed, sell on fear, otherwise hold."""
    if greed > fear and greed > 0.75:
        return 2  # buy
    if fear > greed and fear > 0.75:
        return 0  # sell
    return 1  # hold


def run_pipeline(ticker: str = "AAPL") -> None:
    """Download data, compute features, and execute one heuristic rollout."""
    data = download_price_data(ticker=ticker, start="2020-01-01")
    features = compute_behavioral_signals(data, lookback=20)

    env = TradingEnv(
        features,
        initial_capital=10_000.0,
        max_position=5,
        transaction_cost_pct=0.001,
    )
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    done = False

    while not done:
        fear_level = float(obs[1])
        greed = float(obs[2])

        print("Amygdala activation (fear):", fear_level)
        if fear_level > 0.7:
            print("High fear detected → likely panic selling")

        action = _heuristic_action(fear=fear_level, greed=greed)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    metrics = compute_all_metrics(info["portfolio_value_history"])

    print("Final observation:", obs)
    print("Episode info:", info)
    print(f"Total PnL reward: {total_reward:.4f}")
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    run_pipeline("AAPL")
