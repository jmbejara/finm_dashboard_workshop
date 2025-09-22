# Discussion 1 – Kickoff & Dashboard Preview

## Session Goals
- Experience the end-to-end Streamlit dashboard you will adapt during the workshop.
- Install and launch the `streamlit_ex1` template so you can customize visuals and copy the workflow.
- Clone the [FTSFR](https://github.com/jmbejara/ftsfr) repository and learn how its data pipeline delivers cleaned CRSP extracts.
- Practice basic visual storytelling with WRDS/CRSP data in preparation for deeper forecasting work.

## Prep Checklist (Before We Meet)
- Python 3.11+ available locally (conda or venv is fine).
- Pull this workshop repo and run `pip install -r src/streamlit_ex1/requirements.txt` to get Streamlit, yfinance, Plotly, and the supporting data packages.
- Create a GitHub account if you do not already have one—deployment to Streamlit Community Cloud will request it.
- Review the FTSFR README sections on environment setup and subscriptions (copied in the workshop materials for quick reference).

## Agenda — Day 1

### Segment A · Lecture & Live Demo (12:00 – 1:30 pm)
- Orient to the two-day dashboard goal and the student deliverable rubric.
- Live run-through of `streamlit run src/streamlit_ex1/app.py` highlighting: data sourcing, return tear sheet, forecast preview, and how Streamlit layouts are structured.
- Quick tour of the code (data loader, metrics block, visualization tabs) so students know where to tweak copy, palette, and logic.
- Walkthrough of the Streamlit Community Cloud deployment flow; flag prerequisites (GitHub repo, `requirements.txt`, lightweight secrets handling).
- Short discussion: “What makes a financial dashboard credible?” Collect criteria to revisit later.

### Segment B · Breakout Lab (1:30 – 2:30 pm)
- Task 1: Fork the example app, change the default ticker list, and restyle the line chart (colors, titles, date range defaults).
- Task 2: Add one new summary metric (e.g., rolling Sharpe, worst five drawdowns) to the metrics area.
- Task 3: Deploy locally (`streamlit run ...`) and share a screenshot in Canvas/Slack.
- Optional stretch: Push to GitHub and begin Streamlit Community Cloud deployment (we will finish this after Lecture 2).

### Segment C · Lecture & Guided Walkthrough (2:30 – 4:00 pm)
- Introduce the FTSFR benchmark: motivation, dataset catalog, and how it links to the broader benchmarking conversation (see `reports/draft_ftsfr.tex`).
- Step-by-step: cloning `ftsfr`, creating the `ftsfr` conda environment, copying `.env.example`, and toggling `wrds` access in `subscriptions.toml`.
- Use `doit -f dodo_01_pull.py list` to inspect available tasks; run the CRSP pulls together and inspect the resulting parquet files.
- Demo: load a CRSP parquet file into a notebook, create a minimal histogram and cumulative return plot, and export a CSV for Streamlit consumption.
- Discuss visualization best practices for dashboards (color consistency, context annotations, hover text, layout density).

### Independent Practice (4:00 – 5:00 pm dinner window)
- Encourage students to finish customizing their tear sheet visuals and to explore at least one additional CRSP ticker set for tomorrow’s lab.

## Breakout Reference
- **App path:** `src/streamlit_ex1/app.py`
- **Command:** `streamlit run src/streamlit_ex1/app.py`
- **Key files to edit:** data loader section (`load_price_data`), metric calculations (`calculate_metrics`), Plotly figure definitions in the tabs.
- **Deployment checklist:** push to GitHub, confirm `requirements.txt`, configure secrets (if any) in Streamlit Cloud, and set the entry point to `src/streamlit_ex1/app.py`.

## Deliverables Before Discussion 2
- Local copy of the Streamlit template running with at least one personalized visualization.
- FTSFR repository cloned and able to list DoIt tasks without errors; first CRSP parquet downloaded (store under `_data/` in the `ftsfr` repo or export a copy into this workshop repo’s `data/` folder if preferred).
- Two draft ideas for how forecasting output could integrate into the dashboard (note them in your project journal; we will share tomorrow).
