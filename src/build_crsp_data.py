"""Create the CRSP excerpt consumed by the Streamlit demos.

This module assumes the `doit pull_crsp_data` task has already written the
canonical WRDS pulls (``CRSP_stock_ciz.parquet`` et al.) into ``DATA_DIR``.
We simply load that parquet, compute a small subset of metrics, and write out:

* ``crsp_streamlit_excerpt.csv`` — tidy monthly prices/returns/market cap.
* ``crsp_streamlit_excerpt.parquet`` — the same data in parquet form.
* ``crsp_data_metadata.json`` — tiny metadata blob noting the source and size.

If the parquet is missing (e.g., WRDS credentials unavailable), we fall back to
a deterministic synthetic sample so the workshop can proceed offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from settings import config
from pull_CRSP_Compustat import load_CRSP_stock_ciz

DATA_DIR = Path(config("DATA_DIR"))
DEFAULT_INPUT = DATA_DIR / "CRSP_stock_ciz.parquet"
DEFAULT_OUTPUT = DEFAULT_INPUT
EXCERPT_CSV = DATA_DIR / "crsp_streamlit_excerpt.csv"
EXCERPT_PARQUET = DATA_DIR / "crsp_streamlit_excerpt.parquet"
METADATA_JSON = DATA_DIR / "crsp_data_metadata.json"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_crsp_from_disk() -> Tuple[pd.DataFrame, str]:
    """Load CRSP data produced by the WRDS pull task."""

    if DEFAULT_INPUT.exists():
        return load_CRSP_stock_ciz(data_dir=DATA_DIR), "cached"

    raise FileNotFoundError(
        f"Could not locate {DEFAULT_INPUT}. Run `doit pull_crsp_data` first."
    )


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
        df, source = load_crsp_from_disk()
        if df.empty:
            raise ValueError("CRSP dataframe is empty")
    except Exception as exc:  # pylint: disable=broad-except
        # Provide feedback and fall back to synthetic data to keep the workshop moving.
        print(f"CRSP load failed ({exc}); generating synthetic sample instead.")
        df = generate_synthetic_crsp()
        source = "synthetic"

    excerpt = create_excerpt(df)
    write_outputs(df, excerpt, source)
    print(f"CRSP data ready (source={source}). Excerpt saved to {EXCERPT_CSV}.")
    return source, EXCERPT_CSV


if __name__ == "__main__":
    main()


def _preview_excerpt() -> None:  # pragma: no cover - manual exploration helper
    """Quick-and-dirty snippet for inspecting the excerpt interactively."""

    import matplotlib.pyplot as plt

    if not EXCERPT_CSV.exists():
        print("Excerpt not found. Run `doit create_crsp_excerpt` first.")
        return

    df = pd.read_csv(EXCERPT_CSV, parse_dates=["date"])
    print("Excerpt head:\n", df.head())
    print("Tickers:", df["ticker"].unique())

    pivot = df.pivot(index="date", columns="ticker", values="price")
    pivot.plot(title="Excerpt Prices (Growth of $1 approx)")
    plt.tight_layout()
    plt.show()
