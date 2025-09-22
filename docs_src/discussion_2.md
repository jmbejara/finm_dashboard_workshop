# Discussion 2 – From Diagnostics to Forecast Pipelines

## Session Goals
- Deepen Streamlit skills by wiring CRSP datasets into the template via configuration files or cached parquet extracts.
- Introduce baseline forecasting concepts (naïve, exponential smoothing, rolling factor models) and error metrics.
- Explore the FTSFR forecasting pipeline, focusing on task definitions and how to trigger model runs for equities data.
- Set expectations for overnight work: produce first-pass forecasts and decide how they will surface inside the dashboard.

## Recap & Warm-Up (12:00 – 12:20 pm)
- Group share: screenshots or short Loom clips of customized dashboards.
- Dedicated Q&A on installation snags or Streamlit Cloud deployment attempts.

## Segment A · Lecture (12:20 – 1:10 pm)
- Quick primer on return preprocessing: log vs. simple returns, resampling, handling missing data.
- Forecasting fundamentals: horizon definitions, walk-forward validation, error metrics (MAE, RMSE, MAPE, directional accuracy).
- Benchmark models hierarchy: naive, seasonal naive, exponential smoothing, ARIMA families, and global deep models (per `draft_ftsfr.tex`).
- Reading the FTSFR parquet schema (`id`, `ds`, `y`, optional covariates) and how it simplifies ingestion.

## Segment B · Lab (1:10 – 2:10 pm)
- Hands-on notebook: load CRSP parquet, compute rolling annualized volatility and drawdown, export features to CSV for the Streamlit app.
- Extend the Streamlit template to read from a local parquet instead of live yfinance data (keep the switch configurable so students can toggle sources).
- Add a new tab in the app for “Forecast Inputs” that previews the cleaned dataset columns.

## Segment C · Lecture & Live Coding (2:30 – 3:20 pm)
- FTSFR deep dive: repository map, where DoIt tasks live, and how forecasting jobs reference `datasets.toml` and `models_config.toml`.
- Run `doit -f dodo_02_forecasting.py list` to inspect available jobs; highlight parameters that map to CRSP.
- Demonstrate launching a small forecasting run (e.g., one ticker, short horizon) and inspecting outputs in `_output/forecasting/error_metrics/`.
- Connect outputs back to Streamlit: saving forecasts as CSV and loading them into a Plotly fan chart.

## Segment D · Planning Sprint (3:20 – 4:00 pm)
- Students draft their dashboard forecasting story: which tickers, what horizon, how to explain accuracy.
- Share-out of two volunteers to pressure-test narrative and data flow.

## Homework / Overnight Goals
- Run at least one forecasting job from the FTSFR repo (or, if compute-limited, review the provided sample outputs) and stash the results for integration.
- Sketch a layout for how forecasts will appear in the dashboard (wireframe in Excalidraw/Figma or paper; upload photo to Canvas).
- Optional: attempt Streamlit Cloud deployment again with new datasets, noting any secrets management hurdles to discuss in Discussion 3.
