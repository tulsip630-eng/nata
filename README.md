# Neuro-Adaptive Trading Agent (Initial Prototype)

This repository provides a clean, extensible baseline for a neuro-adaptive trading workflow:

1. **Market data ingestion** via Yahoo Finance (`yfinance`)
2. **Behavioral signal generation** for fear, greed, and stress
3. **Gymnasium-style trading environment** for RL experimentation
4. **PyTorch DQN agent** for buy/hold/sell policy learning

## Project Structure

```text
.
├── pyproject.toml
├── README.md
└── src/
    └── neuro_trader/
        ├── __init__.py
        ├── agents/
        │   ├── __init__.py
        │   ├── dqn_agent.py
        │   └── dual_system_agent.py
        ├── behavioral_signals.py
        ├── data_loader.py
        ├── evaluation_metrics.py
        ├── evaluate.py
        ├── main.py
        ├── trading_env.py
        └── train_dqn.py
```

## Quickstart

```bash
python -m pip install -e .
python -m neuro_trader.main
python -m neuro_trader.train_dqn
python -m neuro_trader.evaluate
```

## Module Overview

- `data_loader.py`
  - `download_price_data(...)` downloads and cleans OHLCV data for a ticker.
- `behavioral_signals.py`
  - `compute_behavioral_signals(...)` adds returns plus normalized (`0..1`) fear, greed, and stress features.
  - **fear** = rolling downside volatility (loss-heavy markets tend to trigger defensive behavior).
  - **greed** = momentum indicator (persistent gains often reinforce reward-seeking/chasing).
  - **stress** = volatility-spike indicator (volatility jumps can increase uncertainty and risk aversion).
- `trading_env.py`
  - `TradingEnv` exposes a Gymnasium environment with `buy/hold/sell` actions.
  - Reward is **change in portfolio value** at each step, net of 0.1% transaction costs by default.
  - Environment tracks portfolio history and position state (`short`, `cash`, `long`).
- `evaluation_metrics.py`
  - Metric functions that take portfolio value series as input.
  - Includes `cumulative_return`, `sharpe_ratio`, `max_drawdown`, and `volatility`.
- `agents/dqn_agent.py`
  - Modular DQN components (`QNetwork`, `ReplayBuffer`, `DQNAgent`, `DQNConfig`).
  - Input: market + behavioral observation vector.
  - Output: Q-values over discrete actions (`sell`, `hold`, `buy`).
- `agents/dual_system_agent.py`
  - Dual-system policy with dynamic emotional/rational weighting.
  - Emotional system weight increases in high volatility; rational weight increases when trend is stable.
- `train_dqn.py`
  - End-to-end training loop for the DQN agent with reusable training metrics.
- `evaluate.py`
  - Train/test evaluation script with unseen 70/30 split, random-strategy benchmark, and buy/hold baseline helper.
  - Saves plots for portfolio value over time, stock price with buy/sell markers, and behavioral signals over time.

## Notes

This baseline is intentionally modular so a future dual-system setup
(e.g., fast reactive policy + slow deliberative policy) can reuse the same environment,
replay buffer, and training interfaces.
