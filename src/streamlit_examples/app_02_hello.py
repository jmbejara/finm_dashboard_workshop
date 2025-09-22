"""Minimal Streamlit app used in the workshop for a quick hello-world demo."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hello Streamlit", page_icon="👋", layout="centered")

st.title("Hello, Streamlit!")
st.write(
    "This tiny app highlights how easy it is to combine markdown, widgets, and plots without any external data."
)

name = st.text_input("What's your name?", "Guest")
st.write(f"Thanks for stopping by, {name}!")

st.divider()

st.subheader("Sine wave quick look")
points = st.slider("Number of points", min_value=30, max_value=200, value=100, step=10)
phase = st.slider("Phase shift", 0.0, 2 * np.pi, 0.0, step=0.1)

x = np.linspace(0, 4 * np.pi, points)
y = np.sin(x + phase)
wave = pd.DataFrame({"x": x, "sine": y})

st.line_chart(wave, x="x", y="sine")

st.caption(
    "Try adjusting the sliders to see how Streamlit rerenders the chart instantly."
)
