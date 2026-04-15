"""Evaluation metrics for trading strategies.

All functions operate on a portfolio-value time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(portfolio_values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Convert input portfolio value sequence to validated pandas Series."""
    series = pd.Series(portfolio_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        raise ValueError("portfolio value series must contain at least two valid points.")
    return series


def portfolio_returns(portfolio_values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Compute step returns from portfolio values."""
    values = _to_series(portfolio_values)
    returns = values.pct_change().dropna()
    if returns.empty:
        raise ValueError("cannot compute returns from provided portfolio values.")
    return returns


def cumulative_return(portfolio_values: pd.Series | np.ndarray | list[float]) -> float:
    """Total cumulative return over the full portfolio value series."""
    values = _to_series(portfolio_values)
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def volatility(
    portfolio_values: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Annualized volatility based on step returns."""
    returns = portfolio_returns(portfolio_values)
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    portfolio_values: pd.Series | np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio computed from step returns."""
    returns = portfolio_returns(portfolio_values)
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = float(excess.std(ddof=1))
    if np.isclose(std, 0.0):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: pd.Series | np.ndarray | list[float]) -> float:
    """Maximum drawdown (minimum of running drawdown series)."""
    values = _to_series(portfolio_values)
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    return float(drawdown.min())


def compute_all_metrics(
    portfolio_values: pd.Series | np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute a standard metric suite for a portfolio-value trajectory."""
    return {
        "cumulative_return": cumulative_return(portfolio_values),
        "sharpe_ratio": sharpe_ratio(
            portfolio_values,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(portfolio_values),
        "volatility": volatility(portfolio_values, periods_per_year=periods_per_year),
    }
