"""Behavioral feature engineering for neuro-adaptive trading.

Signal design in this module maps simple market statistics to behavioral analogs:
- Fear: downside volatility (risk-salience from negative outcomes)
- Greed: momentum (reward-seeking during upward trends)
- Stress: volatility spikes (state of market shock/uncertainty)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1] with safe handling for constant values."""
    finite = series.replace([np.inf, -np.inf], np.nan)
    min_val = float(finite.min(skipna=True))
    max_val = float(finite.max(skipna=True))
    if np.isnan(min_val) or np.isnan(max_val) or np.isclose(max_val - min_val, 0.0):
        return pd.Series(0.0, index=series.index)
    normalized = (finite - min_val) / (max_val - min_val)
    return normalized.fillna(0.0).clip(0.0, 1.0)


def compute_behavioral_signals(
    prices: pd.DataFrame,
    close_col: str = "Close",
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute normalized fear, greed, and stress signals from prices.

    Signal definitions:
    1) fear (downside volatility): rolling std of negative returns only.
       Behavioral interpretation: larger recent downside dispersion increases loss
       sensitivity and defensive behavior.

    2) greed (momentum): lookback-period price momentum.
       Behavioral interpretation: sustained gains encourage reward-seeking,
       trend-chasing, and overconfidence.

    3) stress (volatility spikes): ratio of short-horizon to long-horizon volatility.
       Behavioral interpretation: abrupt volatility regime jumps elevate cognitive
       load and risk aversion.

    All three signals are normalized to the range [0, 1].

    Args:
        prices: DataFrame containing at least a close price column.
        close_col: Name of the close-price column.
        lookback: Rolling window length for statistics.

    Returns:
        DataFrame with original data plus columns:
            ['returns', 'fear', 'greed', 'stress']

    Raises:
        ValueError: If close column does not exist or lookback is invalid.
    """
    if close_col not in prices.columns:
        raise ValueError(f"Expected close column '{close_col}' not found in input data.")
    if lookback < 2:
        raise ValueError("lookback must be >= 2.")

    df = prices.copy()
    close = df[close_col].astype(float)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["returns"] = returns

    # Fear: downside volatility (volatility of losses only).
    downside_returns = returns.where(returns < 0.0, 0.0)
    fear_raw = downside_returns.rolling(lookback, min_periods=2).std().fillna(0.0)

    # Greed: momentum indicator over the lookback horizon.
    greed_raw = close.pct_change(periods=lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Stress: volatility spike proxy from short-vs-long realized volatility.
    short_vol = returns.rolling(max(2, lookback // 2), min_periods=2).std().fillna(0.0)
    long_vol = returns.rolling(lookback, min_periods=2).std().replace(0.0, np.nan)
    stress_raw = (short_vol / long_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["fear"] = _min_max_normalize(fear_raw)
    df["greed"] = _min_max_normalize(greed_raw)
    df["stress"] = _min_max_normalize(stress_raw)

    return df
