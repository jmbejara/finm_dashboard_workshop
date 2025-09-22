"""Streamlit tear sheet demo for the dashboard workshop.

The app highlights:
- Pulling adjusted close prices with yfinance
- Computing return diagnostics and risk metrics
- Visualizing cumulative performance and distribution characteristics
- Generating a light-touch forecast preview with naive log-return dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay

st.set_page_config(
    page_title="Financial Dashboard Tear Sheet",
    layout="wide",
    page_icon="📈",
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "SPY"]
FREQUENCY_MAP = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "M"}
PERIODS_PER_YEAR = {"Daily": 252, "Weekly": 52, "Monthly": 12}


@dataclass
class TearSheetInputs:
    tickers: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    freq_label: str
    forecast_ticker: str
    forecast_horizon: int


@st.cache_data(show_spinner=False)
def load_price_data(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the provided tickers."""
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()

    closes = data["Close"].copy()
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    closes = closes.dropna(how="all")
    closes.index.name = "date"
    return closes


def resample_prices(prices: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    freq = FREQUENCY_MAP[freq_label]
    return prices.resample(freq).last().dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
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
        st.plotly_chart(build_price_chart(prices), use_container_width=True)

    with tab_distribution:
        st.plotly_chart(build_return_distribution(returns), use_container_width=True)
        st.markdown(
            "- Distributions are shown for the selected resampling frequency.\n"
            "- Look for skew, fat tails, and clustering of negative returns when benchmarking models."
        )

    with tab_forecast:
        history = prices[forecast_inputs.forecast_ticker]
        forecast_df = naive_log_forecast(history, forecast_inputs.forecast_horizon)
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


def main() -> None:
    inputs = sidebar_inputs()
    prices = load_price_data(inputs.tickers, inputs.start, inputs.end)

    if prices.empty:
        st.error("No price data returned. Please adjust your date range or tickers.")
        return

    sampled_prices = resample_prices(prices, inputs.freq_label)
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
