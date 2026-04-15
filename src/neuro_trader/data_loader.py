"""Data loading utilities for market data ingestion."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_price_data(
    ticker: str = "AAPL",
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV market data from Yahoo Finance.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        start: Start date in YYYY-MM-DD format.
        end: Optional end date in YYYY-MM-DD format.
        interval: Data interval ("1d", "1h", etc.).
        auto_adjust: Whether to return adjusted prices.

    Returns:
        Cleaned DataFrame indexed by timestamp with OHLCV columns.

    Raises:
        ValueError: If no data is downloaded or required columns are missing.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data returned for ticker={ticker}, start={start}, end={end}."
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    cleaned = data[REQUIRED_COLUMNS].copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index().dropna()
    return cleaned
