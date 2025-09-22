# Discussion 2 – From Diagnostics to Forecast Pipelines

## Session Goals
- Connect the progressive Streamlit apps to newly pulled WRDS data and share customization wins.
- Deliver the benchmarking lecture based on `reports/draft_ftsfr.tex`, covering the motivation, datasets, and evaluation framework.
- Map the forecasting code inside the FTSFR repository so students know where to run experiments after class.
- Set expectations for overnight work: integrate CRSP extracts into the apps and prepare questions for the forecasting lab in Discussion 3.

## Recap & Warm-Up (12:00 – 12:20 pm)
- Group share: screenshots or short Loom clips of customized dashboards.
- Dedicated Q&A on installation snags or Streamlit Cloud deployment attempts.

## Segment A · Lecture (12:20 – 1:20 pm)
- Guided walkthrough of `draft_ftsfr.tex`:
  - Why a financial forecasting benchmark matters and the gaps it fills.
  - Dataset overview (equities, credit, rates, FX, real assets) and how WRDS pulls fit in.
  - Evaluation design: train/test splits, walk-forward protocols, error metrics, and interpretability themes.
  - Model families compared in the paper (naïve baselines, statistical classics, modern global learners).
- Highlight illustrative figures/tables so students can connect terminology to what they will see in code.

## Segment B · Working Session (1:20 – 2:10 pm)
- Quick regroup on app progress: volunteers demo updates to `app_02`, `app_03`, `app_04_crsp`, or `app_01` using last night’s FTSFR CSV exports.
- Hands-on notebook time:
  - Load the CRSP parquet pulled yesterday, compute rolling volatility/drawdown, and save a streamlined CSV for the apps.
  - Wire the CSV into `app_03.py` (replace the bundled sample) and sanity-check the Plotly chart.
- Capture open questions about modeling for the Q&A block after the lecture portion.

## Segment C · Repository Tour & Q&A (2:30 – 3:20 pm)
- Map the FTSFR repository structure: data pulls vs. forecasting vs. reporting modules.
- Show where `datasets.toml`, `models_config.toml`, and the DoIt files live; demonstrate `doit -f dodo_02_forecasting.py list` to enumerate jobs.
- Preview a single forecasting command (no full run required) so students know what to execute overnight.
- Discuss how forecasting outputs (error metrics, prediction CSVs) will feed into `app_01` during Discussion 3.

## Segment D · Planning Sprint (3:20 – 4:00 pm)
- Students outline how they’ll integrate forecasting results: which tickers, what horizon, and how to communicate accuracy vs. uncertainty.
- Share-out from two volunteers to pressure-test narrative and data flow.

## Suggested Next Steps Before Discussion 3
- Run at least one forecasting job from the FTSFR repo (or review the sample outputs if compute-limited) and save the resulting CSVs for use tomorrow.
- Update `app_03.py`, `app_04_crsp.py`, or `app_01.py` to read from your forecast-ready and historical datasets; capture TODOs for integrating visuals.
- Optional: attempt Streamlit Cloud deployment with the updated apps, noting any secrets or environment hurdles for Discussion 3.
