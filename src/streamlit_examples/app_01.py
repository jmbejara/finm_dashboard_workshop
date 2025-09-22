"""Streamlit tear sheet demo for the dashboard workshop.

The app highlights:
- Pulling adjusted close prices with yfinance
- Computing return diagnostics and risk metrics
- Visualizing cumulative performance and distribution characteristics
- Generating a light-touch forecast preview with naive log-return dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay
from pandas_datareader import data as pdr_data

st.set_page_config(
    page_title="Financial Dashboard Tear Sheet",
    layout="wide",
    page_icon="📈",
)

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


def load_sample_prices(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if not SAMPLE_PRICE_PATH.exists():
        return pd.DataFrame()

    sample = pd.read_csv(SAMPLE_PRICE_PATH, parse_dates=["date"], index_col="date")
    sample = sample.loc[(sample.index >= start) & (sample.index <= end)]
    columns = [t for t in tickers if t in sample.columns]
    if not columns:
        return pd.DataFrame()
    return sample[columns].copy()


def try_download_batch(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.DataFrame]:
    symbols = " ".join(tickers)
    if not symbols:
        return None

    try:
        data = yf.download(
            symbols,
            start=start,
            end=end,
            progress=False,
            threads=False,
            group_by="ticker",
        )
    except Exception:
        return None

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(-1):
            close = data.xs("Adj Close", axis=1, level=-1)
        elif "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", axis=1, level=-1)
        else:
            return None
    else:
        if "Adj Close" in data.columns:
            close = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            close = data[["Close"]].copy()
        else:
            return None
        # single ticker response
        col_name = symbols.strip().split(" ")[0]
        close.columns = [col_name]

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    close = close.dropna(how="all")
    return close if not close.empty else None


def fetch_single_ticker(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    try:
        data = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            progress=False,
            threads=False,
        )
    except Exception:
        data = None

    if data is not None and not data.empty:
        candidate_cols = [col for col in ("Adj Close", "Close") if col in data.columns]
        if candidate_cols:
            close = data[candidate_cols[0]].copy()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.Series(close, name=symbol).dropna()
            if not close.empty:
                close.index = pd.to_datetime(close.index)
                if getattr(close.index, "tz", None) is not None:
                    close.index = close.index.tz_localize(None)
                return close

    # Fallback to pandas-datareader if yfinance fails
    try:
        data = pdr_data.get_data_yahoo(symbol, start=start, end=end)
    except Exception:
        return None

    if data is None or data.empty:
        return None

    candidate_cols = [col for col in ("Adj Close", "Close") if col in data.columns]
    if not candidate_cols:
        return None

    close = data[candidate_cols[0]].copy()
    close = pd.Series(close, name=symbol).dropna()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close


def generate_stub_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Create deterministic synthetic prices so the template works offline."""

    index = pd.bdate_range(start, end)
    if index.empty:
        return pd.Series(name=ticker)

    seed = (abs(hash(ticker)) % (2**32)) or 42
    rng = np.random.default_rng(seed)
    mu = 0.0005
    sigma = 0.012
    log_returns = rng.normal(mu, sigma, len(index))
    prices = 100 * np.exp(np.cumsum(log_returns))
    series = pd.Series(prices, index=index, name=ticker)
    return series


@st.cache_data(show_spinner=False)
def load_price_data(
    tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> dict:
    """Download adjusted close prices and capture any tickers that fail."""

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    frames: dict[str, pd.Series] = {}
    failed: List[str] = []

    # Try to load sample data first as a fallback
    sample_prices = load_sample_prices(tickers, start_ts, end_ts)
    if not sample_prices.empty:
        # Use sample data if available
        for col in sample_prices.columns:
            if col in tickers:
                frames[col] = sample_prices[col].dropna()

        missing = [t for t in tickers if t not in frames]
        if missing:
            # Generate synthetic data for missing tickers
            synthetic_frames = [generate_stub_prices(t, start_ts, end_ts) for t in missing]
            for s in synthetic_frames:
                if not s.empty:
                    frames[s.name] = s

        if frames:
            prices = pd.concat(frames.values(), axis=1).sort_index()
            prices.index.name = "date"
            prices = prices.dropna(how="all")
            return {
                "prices": prices.copy(),
                "failed": [],
                "used_sample": True,
                "synthetic": [t for t in missing if t in frames],
            }

    # Try downloading from yfinance if no sample data
    batch = try_download_batch(tickers, start_ts, end_ts)
    if batch is not None:
        for symbol in tickers:
            if symbol in batch.columns:
                ser = batch[symbol].dropna()
                if not ser.empty:
                    frames[symbol] = ser

        missing = [symbol for symbol in tickers if symbol not in frames]
    else:
        missing = list(tickers)

    for symbol in missing:
        series = fetch_single_ticker(symbol, start_ts, end_ts)
        if series is None:
            failed.append(symbol)
            continue
        frames[symbol] = series

    if frames:
        prices = pd.concat(frames.values(), axis=1).sort_index()
        prices.index.name = "date"
        prices = prices.dropna(how="all")
        return {
            "prices": prices.copy(),
            "failed": failed[:],
            "used_sample": False,
            "synthetic": [],
        }

    # If no data from yfinance, try sample data again
    sample_prices = load_sample_prices(tickers, start_ts, end_ts)
    if not sample_prices.empty:
        missing = [t for t in tickers if t not in sample_prices.columns]
        synthetic_frames = [generate_stub_prices(t, start_ts, end_ts) for t in missing]
        combined_frames = [sample_prices] + [s for s in synthetic_frames if not s.empty]
        combined = pd.concat(combined_frames, axis=1).sort_index()
        combined.index.name = "date"
        synthetic_symbols = [s.name for s in synthetic_frames if not s.empty]
        return {
            "prices": combined.copy(),
            "failed": [],
            "used_sample": True,
            "synthetic": synthetic_symbols,
        }

    # As a last resort, synthesize prices for every requested ticker so the UI remains interactive.
    synthetic_frames = [generate_stub_prices(t, start_ts, end_ts) for t in tickers]
    non_empty = [s for s in synthetic_frames if not s.empty]
    if not non_empty:
        return {
            "prices": pd.DataFrame(),
            "failed": list(set(failed)),
            "used_sample": False,
            "synthetic": [],
        }

    combined = pd.concat(non_empty, axis=1)
    combined.index.name = "date"
    return {
        "prices": combined.copy(),
        "failed": [],
        "used_sample": False,
        "synthetic": [s.name for s in non_empty],
    }


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

    ann_return = (1 + returns.mean()) ** periods - 1
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
    )
    metric_df = metric_df.sort_values("Annual Return", ascending=False)
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

    expected_prices = []
    upper_band = []
    lower_band = []
    dates = []

    current_price = last_price
    current_date = clean_series.index[-1]
    for _ in range(horizon):
        current_date = current_date + BDay(1)
        growth = np.exp(mu)
        current_price = current_price * growth
        upper_price = current_price * np.exp(2 * sigma)
        lower_price = current_price * np.exp(-2 * sigma)

        dates.append(current_date)
        expected_prices.append(current_price)
        upper_band.append(upper_price)
        lower_band.append(lower_price)

    forecast_df = pd.DataFrame(
        {
            "expected": expected_prices,
            "upper": upper_band,
            "lower": lower_band,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return forecast_df


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


def render_header(inputs: TearSheetInputs) -> None:
    st.title("From Data to Dashboard – Return Tear Sheet")
    st.caption(
        "Compare equity performance, evaluate risk metrics, and preview a simple forecast. "
        "Customize this template with WRDS/CRSP extracts for the workshop project."
    )
    st.write(
        f"**Universe:** {', '.join(inputs.tickers)} · **Window:** {inputs.start.date()} → {inputs.end.date()} · "
        f"**Frequency:** {inputs.freq_label}"
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
        "Returns and volatility annualized using a {} observation count.".format(
            PERIODS_PER_YEAR[freq_label]
        )
    )


def render_tabs(prices: pd.DataFrame, freq_label: str, forecast_inputs: TearSheetInputs) -> None:
    returns = compute_returns(prices)
    tab_price, tab_distribution, tab_forecast = st.tabs(
        ["Cumulative Returns", "Return Diagnostics", "Forecast Preview"]
    )

    with tab_price:
        st.plotly_chart(build_price_chart(prices), width="stretch")

    with tab_distribution:
        st.plotly_chart(build_return_distribution(returns), width="stretch")
        st.markdown(
            "- Distributions are shown for the selected resampling frequency.\n"
            "- Look for skew, fat tails, and clustering of negative returns when benchmarking models."
        )

    with tab_forecast:
        forecast_ticker = forecast_inputs.forecast_ticker
        if forecast_ticker not in prices.columns:
            st.warning(
                f"Ticker {forecast_ticker} not available in the loaded dataset; showing {prices.columns[0]} instead."
            )
            forecast_ticker = prices.columns[0]
        history = prices[forecast_ticker]
        forecast_df = naive_log_forecast(history, forecast_inputs.forecast_horizon)
        st.plotly_chart(
            build_forecast_chart(history, forecast_df),
            width="stretch",
        )
        st.markdown(
            "This preview uses a simple log-return average with ±2σ bands. Replace this with outputs from the "
            "[FTSFR](https://github.com/jmbejara/ftsfr) forecasting pipeline to showcase richer models."
        )
        if not forecast_df.empty:
            st.dataframe(
                forecast_df.tail(5).style.format("{:.2f}"),
                width="stretch",
            )


def main() -> None:
    inputs = sidebar_inputs()
    load_result = load_price_data(inputs.tickers, inputs.start, inputs.end)

    if load_result["failed"]:
        st.sidebar.warning(
            "Data fetch skipped these symbols: {}".format(
                ", ".join(sorted(load_result["failed"]))
            )
        )

    if load_result["used_sample"]:
        st.sidebar.info(
            "Using bundled sample price data. Connect to the internet or supply WRDS/CRSP extracts for live pulls."
        )

    if load_result["synthetic"]:
        st.sidebar.info(
            "Synthetic placeholder data generated for: {}".format(
                ", ".join(sorted(load_result["synthetic"]))
            )
        )

    if load_result["prices"].empty:
        st.error("No price data returned. Please adjust your date range or tickers.")
        return

    sampled_prices = resample_prices(load_result["prices"], inputs.freq_label)
    if sampled_prices.empty:
        st.error("No prices available after resampling. Try a different frequency.")
        return

    render_header(inputs)
    metrics = calculate_metrics(sampled_prices, inputs.freq_label)
    render_metrics(metrics, inputs.freq_label)
    render_tabs(sampled_prices, inputs.freq_label, inputs)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Workshop Notes")
    st.sidebar.markdown(
        "- Swap the yfinance loader for WRDS/CRSP parquet files.\n"
        "- Persist intermediate data to `data/` and cache heavy computations.\n"
        "- Document your modeling choices inside the app so reviewers know how to interpret the visuals."
    )


if __name__ == "__main__":
    main()
