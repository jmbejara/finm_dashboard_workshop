# FINM September Launch Workshop: From Data to Dashboard

This repository accompanies the **From Data to Dashboard: Building Interactive Financial Visualizations in Python** workshop. It contains Streamlit demos, automation scripts, and Sphinx documentation that guide students from a simple “hello world” app to a CRSP-powered dashboard informed by the Financial Time-Series Forecasting Repository (FTSFR).

## Documentation
The full workshop guide lives at [https://jeremybejarano.com/finm_dashboard_workshop/](https://jeremybejarano.com/finm_dashboard_workshop/). The code to build this documentation is in this repo and you can build eveything locally by running `doit` (see below) and opening the generated pages under `docs/`.

## Required Software & Accounts
- **Python 3.11+ (Anaconda recommended)** – Install the [Anaconda distribution](https://www.anaconda.com/products/distribution) and verify `streamlit hello` runs locally.
- **Editor** – VS Code with the Python/Jupyter extensions or your preferred IDE.
- **Git & GitHub** – Install [Git](https://git-scm.com/downloads) and make sure you have a GitHub account for sharing/deployment.
- **WRDS credentials** – Request access early and test the connection:
  ```python
  import wrds
  db = wrds.Connection(wrds_username="your_username")
  ```
  Select `Y` if prompted to create a `.pgpass` file.

## Quick Start
```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Pull (or synthesize) CRSP data and build the docs
doit

# Launch the CRSP dashboard demo
streamlit run src/streamlit_examples/app_04_crsp.py
```

The default `doit` run performs three tasks:
1. Attempts to pull CRSP/Compustat data via WRDS (falls back to a synthetic sample if credentials are unavailable).
2. Generates a tidy excerpt for the Streamlit apps.
3. Builds the Sphinx site into `_docs/build/html/` and mirrors it into `docs/`.

## Streamlit Examples
- `src/streamlit_examples/app_02_hello.py` – minimal hello world (text input + sine wave).
- `src/streamlit_examples/app_03.py` – sidebar navigation, cached CSV loading, Plotly visualization.
- `src/streamlit_examples/app_01.py` – tear-sheet layout (yfinance with offline fallbacks).
- `src/streamlit_examples/app_04_crsp.py` – same layout powered by the CRSP excerpt generated via `doit`.

Customize the apps during the workshop and use them as scaffolds for your final dashboard.

## Automation & Data
- `dodo.py` – defines the `doit` tasks (CRSP pull, excerpt creation, Sphinx build/publish).
- `src/build_crsp_data.py` – orchestrates the WRDS pull or synthetic fallback and writes the Streamlit-ready CSV/parquet.
- Data outputs live under `_data/` (both the raw pulls and the Streamlit excerpt/metadata).

## FTSFR Integration
The workshop leans on the external [FTSFR](https://github.com/jmbejara/ftsfr) repository for larger WRDS/Bloomberg datasets and forecasting utilities. See `docs_src/intro_to_ftsfr_data.md` for setup instructions, environment configuration, and recommended tasks (`dodo_01_pull.py`, `dodo_02_forecasting.py`, etc.). Once you pull datasets there, point the Streamlit loaders in this repo to the exported parquet/CSV files.

## Need Help?
Reach out to Jeremy Bejarano (jbejarano@uchicago.edu). Issues and pull requests are welcome if you spot improvements for the workshop materials.

## Misc
`ruff format . && ruff check --fix --show-fixes . && ruff check .`
