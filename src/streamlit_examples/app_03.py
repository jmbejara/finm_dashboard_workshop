"""Intermediate Streamlit example that layers in data loading and navigation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path(__file__).with_name("sample_prices.csv")

st.set_page_config(page_title="Streamlit Example 03", page_icon="📊", layout="wide")

st.title("Exploring Sample Price Data")
st.write(
    "This app introduces a simple navigation pattern and loads a CSV file bundled with the workshop "
    "materials. Use it as a stepping stone between the hello-world example and the full dashboard demo."
)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Could not find sample_prices.csv next to this script.")
        return pd.DataFrame()

    data = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
    data.index.name = "date"
    return data


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
