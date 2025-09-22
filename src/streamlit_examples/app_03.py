"""Intermediate Streamlit example that layers in data loading and navigation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Streamlit Example 03", page_icon="📊", layout="wide")

st.title("Exploring Sample Price Data")
st.write(
    "This app introduces a simple navigation pattern and loads price data from Yahoo Finance. "
    "Use it as a stepping stone between the hello-world example and the full dashboard demo."
)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        closes = data["Close"].copy()
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        closes.index.name = "date"
        closes = closes.dropna(how="all")
        return closes
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()


def render_overview(prices: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    st.write(
        "Pick a subset of tickers and a date range to preview the underlying price data."
    )
    tickers = st.multiselect(
        "Tickers",
        options=list(prices.columns),
        default=list(prices.columns[:2]),
    )
    if not tickers:
        st.info("Select at least one ticker to continue.")
        return

    start, end = st.slider(
        "Date range",
        min_value=prices.index.min().to_pydatetime(),
        max_value=prices.index.max().to_pydatetime(),
        value=(prices.index.min().to_pydatetime(), prices.index.max().to_pydatetime()),
        format="YYYY-MM-DD",
    )
    filtered = prices.loc[start:end, tickers]

    st.dataframe(filtered.tail(10))

    summary = filtered.pct_change(fill_method=None).agg(["mean", "std", "min", "max"]).T
    summary.columns = ["Avg Return", "Volatility", "Worst", "Best"]
    st.markdown("### Return Summary (daily pct change)")
    st.dataframe(summary.style.format("{:.2%}"))


def render_visuals(prices: pd.DataFrame) -> None:
    st.subheader("Interactive Visualization")
    ticker = st.selectbox("Ticker", options=list(prices.columns))
    subset = prices[[ticker]].reset_index()
    subset["normalized"] = subset[ticker] / subset[ticker].iloc[0]

    line_chart = px.line(
        subset,
        x="date",
        y="normalized",
        title=f"Growth of $1 invested in {ticker}",
        labels={"normalized": "Growth", "date": "Date"},
    )
    line_chart.update_layout(template="plotly_white")
    st.plotly_chart(line_chart, width="stretch")

    st.markdown(
        "Try connecting this pattern to WRDS/CRSP extracts—swap the CSV for your own data file and "
        "reuse the widgets to explore new tickers."
    )


prices = load_prices()
if prices.empty:
    st.stop()

page = st.sidebar.radio(
    "Choose a section",
    options=["Overview", "Visualizations"],
    index=0,
)

if page == "Overview":
    render_overview(prices)
else:
    render_visuals(prices)
