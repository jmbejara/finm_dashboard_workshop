# Intro to the FTSFR Repository

We will use the **Financial Time-Series Forecasting Repository (FTSFR)** as our source of standardized WRDS/Bloomberg datasets and forecasting utilities. Clone it ahead of Discussion 2 so we can explore the benchmark together.

## Why FTSFR Matters
- Provides reproducible pulls for multiple asset classes (equity, credit, rates, FX, real assets).
- Standardizes each dataset into a consistent panel format (`id`, `ds`, `y`, plus optional covariates) so you can swap models without rewriting ingestion code.
- Includes automation scripts (`dodo_*.py`) that mirror the pipeline described in `reports/draft_ftsfr.tex`.
- Ships forecasting jobs and evaluation helpers that compare classical models with modern global learners.

## Getting Set Up
```bash
git clone https://github.com/jmbejara/ftsfr.git
cd ftsfr
conda create -n ftsfr python=3.12
conda activate ftsfr
pip install -r requirements.txt
# Optional forecasting extras
pip install -r requirements_forecasting.txt
```
Optional tools:
- Bloomberg users: `python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`
- LaTeX build: install TeX Live (or your preferred TeX distribution) if you plan to compile the paper.

Verify the environment:
```bash
doit --version
```

## Configure Credentials
1. Copy `.env.example` to `.env` and populate secrets such as `WRDS_USERNAME`.
2. Edit `subscriptions.toml` to reflect what data sources and models you can access. Example:
   ```toml
   [cache]
   use_cache = true

   [data_sources]
   bloomberg = false
   wrds = true
   public = true

   [models]
   darts_tcn = true
   prophet = false
   ```
   Set `use_cache = false` if you need a fresh pull; disable sources you can’t reach.

## Core Workflows
1. **Pull Bloomberg data (optional)**
   ```bash
    doit -f dodo_00_pull_bloomberg.py
   ```
   Only run this on a machine with the Bloomberg Desktop API configured and `bloomberg = true` in `subscriptions.toml`.

2. **Pull WRDS/Public datasets**
   ```bash
    doit -f dodo_01_pull.py
   ```
   Targets arrive under `_data/<dataset>/` as parquet + metadata.

3. **Run forecasting jobs**
   ```bash
    doit -f dodo_02_forecasting.py
   ```
   Generates prediction CSVs, error metrics, and timing logs in `_output/forecasting/`.

4. **Build documentation/paper (optional)**
   ```bash
    doit -f dodo_03_paper.py
   ```
   Rebuilds the static site and compiles `reports/draft_ftsfr.tex`.

## Navigating the Repo
- `src/` – data modules (`pull_*` scripts), forecasting code, and utilities by asset class.
- `_data/` – raw/processed datasets produced by the pulls.
- `_output/` – forecasts, diagnostics, summary tables, timing logs.
- `datasets.toml` – catalog of dataset names, frequencies, and dependencies.
- `reports/` – LaTeX draft plus figures that mirror the Sphinx docs.

## Working with Outputs
- Error metrics: `_output/forecasting/error_metrics/<dataset>/<model>.csv`
- Timing logs: `_output/forecasting/timing/<model>/<dataset>_timing.csv`
- Prediction series: `_output/forecasting/predictions/<dataset>/<model>.parquet`
- Summary dashboards (if present): `_output/forecasting/summary/`

After a pull, copy the relevant parquet/CSV files into this workshop repo (or point the Streamlit loaders directly at the FTSFR paths) to keep your dashboards up to date.

## Recommended Prep Before Discussion 2
- Clone the repo, set up the environment, and run `doit -f dodo_01_pull.py` with `wrds = true` to ensure your credentials work.
- Skim `reports/draft_ftsfr.tex` to familiarize yourself with the benchmark’s motivation and terminology—we’ll refer back to it during the lecture.
- Note any questions about the pipeline or models so we can address them during the walkthrough.
