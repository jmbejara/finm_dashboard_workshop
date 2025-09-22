# Discussion 2 – Benchmarking & Forecasting

This session pivots from dashboards to the theory and tooling behind the Financial Time-Series Forecasting Repository (FTSFR).

## Slide Talk: Why a Financial Forecasting Benchmark?
- We’ll walk through the slides in [slides_ftsfr.pdf](assets/slides_ftsfr.pdf).
- Topics include: the motivation for a dedicated finance benchmark, dataset design, train/test protocol, and how the team evaluates competing models.
- Bring questions about asset classes, evaluation metrics, or how the benchmark might align with your own research interests.

## Repository Walkthrough
- Tour the FTSFR repo structure (`src/`, `_data/`, `_output/`, `datasets.toml`, `subscriptions.toml`).
- Review environment setup, `.env` credentials, and the key `doit` scripts you’ll use (`dodo_01_pull.py`, `dodo_02_forecasting.py`).
- Highlight where forecasting jobs drop their predictions and error metrics so you can wire them into the Streamlit apps.

## Hands-On Forecasting
- Kick off at least one forecasting job (or use the provided sample outputs if compute is limited).
- Inspect the generated CSV/parquet files and verify the schema matches what your dashboard expects.
- Capture any TODOs needed to plug the forecasts into `app_01.py` or `app_04_crsp.py` before Discussion 3.

## Suggested Next Steps
- Run a full forecasting batch overnight and stash the outputs for integration.
- Start weaving forecasts into your dashboard—new tabs, comparison charts, or narrative callouts.
- Note any blockers (data access, compute, modeling questions) so we can troubleshoot them in the next session.
