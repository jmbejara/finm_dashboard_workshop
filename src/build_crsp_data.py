"""Utility script to pull CRSP data (with graceful fallbacks) and prepare a
streamlit-friendly excerpt.

The script attempts to use the WRDS connection defined in
`src/load_CRSP_Compustat.py`. If the pull fails—because credentials are
missing or the connection is blocked—it generates a synthetic sample so that the
workshop exercises can proceed offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import config
from load_CRSP_Compustat import pull_CRSP_stock_ciz

DATA_DIR = Path(config.DATA_DIR)
PULLED_DIR = DATA_DIR / "pulled"
DERIVED_DIR = DATA_DIR / "derived"
DEFAULT_OUTPUT = PULLED_DIR / "CRSP_stock_ciz.parquet"
EXCERPT_CSV = DERIVED_DIR / "crsp_streamlit_excerpt.csv"
EXCERPT_PARQUET = DERIVED_DIR / "crsp_streamlit_excerpt.parquet"
METADATA_JSON = DERIVED_DIR / "crsp_data_metadata.json"


def ensure_directories() -> None:
    PULLED_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_live_crsp() -> pd.DataFrame:
    """Attempt to pull CRSP monthly stock data via WRDS."""
    return pull_CRSP_stock_ciz(wrds_username=config.WRDS_USERNAME)


def generate_synthetic_crsp() -> pd.DataFrame:
    """Create a synthetic CRSP-like dataset for offline demos."""
    rng = np.random.default_rng(42)
    permnos = [10001, 10002, 10003]
    permcos = [5001, 5002, 5003]
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")

    frames = []
    for permno, permco in zip(permnos, permcos):
        base_price = rng.uniform(20, 150)
        shocks = rng.normal(loc=0.01, scale=0.05, size=len(dates))
        prices = base_price * np.exp(np.cumsum(shocks))
        returns = np.concatenate([[np.nan], np.diff(prices) / prices[:-1]])
        shrout = rng.integers(50_000, 200_000, size=len(dates))

        frames.append(
            pd.DataFrame(
                {
                    "permno": permno,
                    "permco": permco,
                    "mthcaldt": dates,
                    "mthret": returns,
                    "mthprc": prices,
                    "shrout": shrout,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["mthret"] = df["mthret"].fillna(0.0)
    return df


def create_excerpt(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy excerpt suitable for Streamlit visualizations."""
    required_cols = {"permno", "mthcaldt", "mthprc", "mthret", "shrout"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CRSP dataframe is missing required columns: {missing}")

    excerpt = df.copy()
    excerpt = excerpt.dropna(subset=["mthcaldt", "mthprc"])
    excerpt["mthcaldt"] = pd.to_datetime(excerpt["mthcaldt"])
    excerpt["price"] = excerpt["mthprc"].abs()
    excerpt["return"] = excerpt["mthret"].astype(float)
    excerpt["market_cap"] = excerpt["price"] * excerpt["shrout"].astype(float)
    excerpt["ticker"] = excerpt["permno"].astype(str)

    # Keep a manageable subset: pick the top 5 permnos by ending market cap
    latest = excerpt.sort_values("mthcaldt").groupby("permno").tail(1)
    keep_permnos = (
        latest.sort_values("market_cap", ascending=False)
        .head(5)["permno"]
        .astype(int)
        .tolist()
    )
    excerpt = excerpt[excerpt["permno"].isin(keep_permnos)]

    excerpt = excerpt[
        ["mthcaldt", "ticker", "price", "return", "market_cap", "permno", "permco"]
    ].rename(columns={"mthcaldt": "date"})

    return excerpt.sort_values(["ticker", "date"])


def write_outputs(df: pd.DataFrame, excerpt: pd.DataFrame, source: str) -> None:
    ensure_directories()
    df.to_parquet(DEFAULT_OUTPUT)
    excerpt.to_parquet(EXCERPT_PARQUET)
    excerpt.to_csv(EXCERPT_CSV, index=False)
    METADATA_JSON.write_text(
        json.dumps(
            {
                "source": source,
                "rows": int(df.shape[0]),
                "unique_permnos": int(df["permno"].nunique()),
            },
            indent=2,
        )
    )


def main() -> Tuple[str, Path]:
    ensure_directories()
    try:
        df = fetch_live_crsp()
        if df.empty:
            raise ValueError("Fetched CRSP dataframe is empty")
        source = "wrds"
    except Exception as exc:  # pylint: disable=broad-except
        # Provide feedback and fall back to synthetic data to keep the workshop moving.
        print(f"CRSP pull failed ({exc}); generating synthetic sample instead.")
        df = generate_synthetic_crsp()
        source = "synthetic"

    excerpt = create_excerpt(df)
    write_outputs(df, excerpt, source)
    print(f"CRSP data ready (source={source}). Excerpt saved to {EXCERPT_CSV}.")
    return source, EXCERPT_CSV


if __name__ == "__main__":
    main()
