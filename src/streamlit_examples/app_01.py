"""Streamlit tear sheet demo for the workshop.

Pedagogical focus
-----------------
* Demonstrate how to pull prices with yfinance in just a few lines.
* Compute a handful of reusable performance metrics.
* Show how Streamlit + Plotly can become a polished dashboard.

Data flow
---------
`load_price_data` tries (in order):
1. Live downloads from Yahoo Finance.
2. The bundled CSV (`sample_prices.csv`).
3. A deterministic synthetic series so the app still renders offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial Dashboard Tear Sheet",
    layout="wide",
    page_icon="📈",
)

# -----------------------------------------------------------------------------
# Constants & dataclasses
# -----------------------------------------------------------------------------

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "SPY"]
FREQUENCY_MAP = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "M"}
PERIODS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}
SAMPLE_PRICE_PATH = Path(__file__).with_name("sample_prices.csv")


@dataclass
class TearSheetInputs:
    tickers: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    freq_label: str
    forecast_ticker: str
    forecast_horizon: int


@dataclass
class PriceBundle:
    prices: pd.DataFrame
    failed: List[str]
    source: str  # "live", "sample", "synthetic", or "mixed"
    synthetic: List[str]


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def _download_yfinance_prices(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Download adjusted close prices; return empty frame on failure."""

    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            progress=False,
            group_by="ticker",
        )
    except Exception:  # pragma: no cover - network errors
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex when requesting multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        for level in ("Adj Close", "Close"):
            if level in raw.columns.get_level_values(-1):
                close = raw.xs(level, axis=1, level=-1)
                break
        else:
            return pd.DataFrame()
    else:  # single ticker
        column = "Adj Close" if "Adj Close" in raw else "Close"
        close = raw[[column]].copy()
        close.columns = [list(tickers)[0]]

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    return close.dropna(how="all")


def _load_sample_prices(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if not SAMPLE_PRICE_PATH.exists():
        return pd.DataFrame()

    sample = pd.read_csv(SAMPLE_PRICE_PATH, parse_dates=["date"], index_col="date")
    sample = sample.loc[(sample.index >= start) & (sample.index <= end)]
    keep = [ticker for ticker in tickers if ticker in sample.columns]
    return sample[keep].copy() if keep else pd.DataFrame()


def _generate_stub_prices(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    index = pd.bdate_range(start, end)
    if index.empty:
        return pd.Series(name=ticker)

    seed = (abs(hash(ticker)) % (2**32)) or 42
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0005, 0.012, len(index))
    prices = 100 * np.exp(np.cumsum(shocks))
    return pd.Series(prices, index=index, name=ticker)


@st.cache_data(show_spinner=False)
def load_price_data(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> PriceBundle:
    """Return price data plus metadata about fallbacks used."""

    tickers = list(tickers) or [DEFAULT_TICKERS[0]]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    frames: dict[str, pd.Series] = {}
    failed: List[str] = []
    synthetic_used: List[str] = []
    source = "live"

    # 1. Try live data
    live_prices = _download_yfinance_prices(tickers, start, end)
    if not live_prices.empty:
        for ticker in live_prices.columns:
            frames[ticker] = live_prices[ticker].dropna()
    else:
        failed.extend(tickers)

    # 2. Fill gaps with sample data if available
    missing = [ticker for ticker in tickers if ticker not in frames]
    if missing:
        sample = _load_sample_prices(missing, start, end)
        if not sample.empty:
            for ticker in sample.columns:
                frames[ticker] = sample[ticker].dropna()
            source = "mixed" if frames and len(frames) > len(sample.columns) else "sample"
        missing = [ticker for ticker in tickers if ticker not in frames]

    # 3. Generate deterministic synthetic data for anything left
    if missing:
        for ticker in missing:
            frames[ticker] = _generate_stub_prices(ticker, start, end)
            synthetic_used.append(ticker)
        source = "mixed" if frames and len(frames) > len(missing) else "synthetic"

    prices = pd.concat(frames.values(), axis=1).sort_index()
    prices.columns = list(frames.keys())

    return PriceBundle(prices=prices, failed=failed, source=source, synthetic=synthetic_used)


# -----------------------------------------------------------------------------
# Metrics and charts
# -----------------------------------------------------------------------------


def resample_prices(prices: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    freq = FREQUENCY_MAP[freq_label]
    return prices.resample(freq).last().dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns.index.name = "date"
    return returns


def calculate_metrics(prices: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    returns = compute_returns(prices)
    periods = PERIODS_PER_YEAR[freq_label]

    ann_return = (1 + returns).prod() ** (periods / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(periods)
    sharpe = ann_return / ann_vol.replace({0: np.nan})

    normalized = prices / prices.iloc[0]
    rolling_max = normalized.cummax()
    drawdowns = normalized / rolling_max - 1
    max_drawdown = drawdowns.min()

    metric_df = pd.DataFrame(
        {
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        }
    ).sort_values("Annual Return", ascending=False)
    return metric_df


def build_price_chart(prices: pd.DataFrame) -> go.Figure:
    normalized = prices / prices.iloc[0]
    fig = go.Figure()
    for column in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(
        title="Cumulative Performance (Indexed to 100)",
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        template="plotly_white",
    )
    return fig


def build_return_distribution(returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in returns.columns:
        fig.add_trace(
            go.Histogram(
                x=returns[column],
                name=column,
                opacity=0.6,
                nbinsx=50,
            )
        )
    fig.update_layout(
        title="Distribution of Period Returns",
        xaxis_title="Return",
        yaxis_title="Frequency",
        barmode="overlay",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def naive_log_forecast(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Simple log-return-based forecast with +/-2 sigma envelopes."""

    clean_series = series.dropna()
    if clean_series.empty or horizon <= 0:
        return pd.DataFrame()

    log_returns = np.log(clean_series / clean_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = float(clean_series.iloc[-1])

    dates = []
    expected_prices = []
    upper_band = []
    lower_band = []

    current_price = last_price
    current_date = clean_series.index[-1]
    for _ in range(horizon):
        current_date = current_date + BDay(1)
        current_price *= np.exp(mu)
        expected_prices.append(current_price)
        upper_band.append(current_price * np.exp(2 * sigma))
        lower_band.append(current_price * np.exp(-2 * sigma))
        dates.append(current_date)

    return pd.DataFrame(
        {
            "expected": expected_prices,
            "upper": upper_band,
            "lower": lower_band,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def build_forecast_chart(history: pd.Series, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history,
            mode="lines",
            name="Historical",
            line=dict(color="#2a5ada"),
        )
    )
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["upper"],
                mode="lines",
                name="Upper band",
                line=dict(color="#89c2d9", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["lower"],
                mode="lines",
                name="Lower band",
                line=dict(color="#fec89a", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["expected"],
                mode="lines",
                name="Expected path",
                line=dict(color="#f3722c"),
            )
        )
    fig.update_layout(
        title="Forecast Preview",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# -----------------------------------------------------------------------------
# Sidebar and layout
# -----------------------------------------------------------------------------


def sidebar_inputs() -> TearSheetInputs:
    st.sidebar.header("Configuration")

    tickers = st.sidebar.multiselect(
        "Select tickers",
        options=DEFAULT_TICKERS,
        default=["AAPL", "MSFT", "SPY"],
        help="Pick up to five symbols to compare",
    )

    if not tickers:
        tickers = [DEFAULT_TICKERS[0]]

    start = st.sidebar.date_input("Start date", pd.Timestamp.today() - pd.DateOffset(years=5))
    end = st.sidebar.date_input("End date", pd.Timestamp.today())
    freq_label = st.sidebar.radio("Resample frequency", list(FREQUENCY_MAP.keys()), index=0)

    forecast_ticker = st.sidebar.selectbox(
        "Ticker for forecast preview",
        options=tickers,
        index=0,
    )
    forecast_horizon = st.sidebar.slider(
        "Forecast horizon (business days)",
        min_value=10,
        max_value=90,
        value=30,
        step=5,
    )

    return TearSheetInputs(
        tickers=tickers,
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
        freq_label=freq_label,
        forecast_ticker=forecast_ticker,
        forecast_horizon=forecast_horizon,
    )


def render_header(bundle: PriceBundle, inputs: TearSheetInputs) -> None:
    st.title("From Data to Dashboard – Return Tear Sheet")
    st.caption(
        "Compare equity performance, evaluate risk metrics, and preview a simple forecast. "
        "Customize this template with WRDS/CRSP extracts for the workshop project."
    )
    st.write(
        f"**Universe:** {', '.join(inputs.tickers)} · **Window:** {inputs.start.date()} → {inputs.end.date()} · "
        f"**Frequency:** {inputs.freq_label}"
    )

    if bundle.failed:
        st.sidebar.warning(
            "Skipped tickers (no live data): {}".format(", ".join(sorted(bundle.failed)))
        )
    if bundle.source != "live":
        message = {
            "sample": "Using bundled sample data.",
            "synthetic": "Using synthetic data for all tickers.",
            "mixed": "Using a mix of live/sample/synthetic data.",
        }.get(bundle.source, "Data fallback in effect.")
        st.sidebar.info(message)
    if bundle.synthetic:
        st.sidebar.info(
            "Synthetic series generated for: {}".format(", ".join(bundle.synthetic))
        )


def render_metrics(metrics: pd.DataFrame, freq_label: str) -> None:
    st.subheader("Summary Metrics")
    st.dataframe(
        metrics.style.format(
            {
                "Annual Return": "{:.1%}",
                "Annual Volatility": "{:.1%}",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown": "{:.1%}",
            }
        )
    )
    st.caption(
        "Returns and volatility annualized using {} observations per year.".format(
            PERIODS_PER_YEAR[freq_label]
        )
    )


def render_tabs(prices: pd.DataFrame, freq_label: str, inputs: TearSheetInputs) -> None:
    returns = compute_returns(prices)
    tab_price, tab_distribution, tab_forecast = st.tabs(
        ["Cumulative Returns", "Return Diagnostics", "Forecast Preview"]
    )

    with tab_price:
        st.plotly_chart(build_price_chart(prices), use_container_width=True)

    with tab_distribution:
        st.plotly_chart(build_return_distribution(returns), use_container_width=True)
        st.markdown(
            "- Distributions are shown for the selected resampling frequency.\n"
            "- Look for skew, fat tails, and clustering of negative returns when benchmarking models."
        )

    with tab_forecast:
        ticker = inputs.forecast_ticker
        if ticker not in prices.columns:
            ticker = prices.columns[0]
            st.warning(
                f"Ticker {inputs.forecast_ticker} not in dataset; showing {ticker} instead."
            )
        history = prices[ticker]
        forecast_df = naive_log_forecast(history, inputs.forecast_horizon)
        st.plotly_chart(
            build_forecast_chart(history, forecast_df),
            use_container_width=True,
        )
        st.markdown(
            "This preview uses a simple log-return average with ±2σ bands. Replace this with outputs from the "
            "[FTSFR](https://github.com/jmbejara/ftsfr) forecasting pipeline to showcase richer models."
        )
        if not forecast_df.empty:
            st.dataframe(
                forecast_df.tail(5).style.format("{:.2f}"),
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# App entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Draw the Streamlit layout in three phases (sidebar, metrics, charts)."""

    # Sidebar: gather tickers/date range from the user
    inputs = sidebar_inputs()

    # Data: load prices (live/sample/synthetic) and resample to requested frequency
    bundle = load_price_data(inputs.tickers, inputs.start, inputs.end)
    if bundle.prices.empty:
        st.error("No price data returned. Please adjust your date range or tickers.")
        return
    sampled_prices = resample_prices(bundle.prices, inputs.freq_label)
    if sampled_prices.empty:
        st.error("No prices available after resampling. Try a different frequency.")
        return

    # Header + metrics
    render_header(bundle, inputs)
    metrics = calculate_metrics(sampled_prices, inputs.freq_label)
    render_metrics(metrics, inputs.freq_label)

    # Tabs: cumulative returns, diagnostics, simple forecast preview
    render_tabs(sampled_prices, inputs.freq_label, inputs)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Workshop Notes")
    st.sidebar.markdown(
        "- Swap the yfinance loader for WRDS/CRSP parquet files.\n"
        "- Persist intermediate data to `_data/` and cache heavy computations.\n"
        "- Document your modeling choices inside the app so reviewers know how to interpret the visuals."
    )


if __name__ == "__main__":
    main()
