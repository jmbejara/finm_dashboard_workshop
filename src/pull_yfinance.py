"""Fetch benchmark price data from Yahoo Finance for the Streamlit demos.

The script downloads adjusted close prices for a small set of tickers and stores
both the raw price panel and a tidy CSV used by `app_01.py` as the offline
fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
yf.__version__
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PRICES_PARQUET = DATA_DIR / "yfinance_prices.parquet"
PRICES_CSV = DATA_DIR / "yfinance_prices.csv"
DEFAULT_TICKERS: Iterable[str] = ["AAPL", "MSFT", "SPY"]


def fetch_prices(tickers: Iterable[str] = DEFAULT_TICKERS) -> pd.DataFrame:
    data = yf.download(
        tickers=list(tickers),
        start="2020-01-01",
        end=None,
        auto_adjust=True,
        progress=False,
    )
    closes = data["Close"].copy()
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    closes.index.name = "date"
    closes = closes.dropna(how="all")
    return closes


def write_outputs(prices: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PARQUET)
    prices.to_csv(PRICES_CSV)


def main() -> None:
    prices = fetch_prices(DEFAULT_TICKERS)
    write_outputs(prices)
    print(f"Saved Yahoo Finance prices to {PRICES_PARQUET} and {PRICES_CSV}")
    print("Copy PRICES_CSV to src/streamlit_examples/sample_prices.csv if you wish to update the fallback file.")


if __name__ == "__main__":
    main()
