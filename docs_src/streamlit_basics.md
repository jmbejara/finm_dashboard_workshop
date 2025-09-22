# Streamlit App Progression

We use four small apps to highlight how quickly Streamlit scales from a single chart to a data-rich dashboard. Run them in order to see how each layer adds new concepts.

## Environment
```bash
pip install -r src/streamlit_examples/requirements.txt
```
Each app can be launched with `streamlit run <path-to-app.py>`. Use a new terminal tab for each run so you can compare layouts side-by-side.

## App 02 – Hello Streamlit (`src/streamlit_examples/app_02_hello.py`)
- Minimal “hello world” example: text input, sliders, and a sine-wave chart generated with NumPy/Pandas.
- Purpose: show the Streamlit rerender loop and demonstrate how little boilerplate is needed.
- Try customizing: change the default greeting, swap the sine wave for a cosine wave, or add a checkbox that toggles grid-lines.

## App 03 – Intermediate Navigation (`src/streamlit_examples/app_03.py`)
- Introduces cached CSV loading (`@st.cache_data`), sidebar navigation, and Plotly Express visuals.
- Defaults to `sample_prices.csv`, but it’s designed to load any WRDS/CRSP export—replace the file with your own data slice when you’re ready.
- Try customizing: add a second tab for distribution plots, expand the summary table with Sharpe/Sortino ratios, or link to the Streamlit Cloud deployment instructions.

## App 01 – Tear Sheet (`src/streamlit_examples/app_01.py`)
- Full dashboard experience: multi-ticker selection, annualized metrics, cumulative performance, histogram of returns, and a naïve forecast preview.
- Uses `yfinance` by default but gracefully falls back to bundled sample data or synthetic prices, so you can experiment offline.
- Try customizing: connect the loader to your CRSP extract, restyle the Plotly charts, or replace the log-return forecast with an FTSFR-generated CSV.

## App 04 – CRSP Snapshot (`src/streamlit_examples/app_04_crsp.py`)
- Mirrors the tear-sheet layout but reads the excerpt created by `doit pull_crsp_data` (`data/derived/crsp_streamlit_excerpt.csv`).
- Ideal for demonstrating how WRDS pulls flow through the pipeline: rerun `doit` and refresh the app to see new tickers or time ranges.
- Try customizing: surface additional CRSP attributes (e.g., shares outstanding, market cap rankings) or add narrative callouts describing the trends you observe.

## Tips for Iteration
- Use the sidebar to surface configuration switches instead of scattering inputs throughout the layout.
- Cache aggressively (`st.cache_data`) when loading large CSV/parquet files so Streamlit isn’t re-reading data with every interaction.
- Keep charts and metrics backed by small helper functions—this makes it easier to port the logic into your final project.
- When you’re ready to deploy, ensure `requirements.txt` contains every package you import and that you use relative paths (or environment variables) for data.

These scaffolds are intentionally lightweight; fork them, experiment, and mix/match components as you plan your own dashboard.
