# Automation with `doit`

This repository uses [`doit`](https://pydoit.org/) to glue together data pulls, documentation builds, and other repeatable tasks. Think of it as a light-weight, Python-friendly alternative to a Makefile. Run this before customizing the Streamlit apps so the CRSP excerpt exists.

## Installation
```bash
pip install -r requirements.txt
```
This installs `doit` alongside the other packages used in the workshop.

## Inspecting Available Tasks
```bash
doit list
```
Key tasks:
- `pull_crsp_data` – runs `src/build_crsp_data.py`, which attempts a WRDS pull and falls back to synthetic CRSP-style data when necessary.
- `show_crsp_excerpt_info` – prints a quick summary after the pull completes.
- `build_docs` – builds the Sphinx site into `_docs/build/html/`.
- `publish_docs` – copies the HTML into `docs/` (suitable for GitHub Pages).

`doit` runs the default task list defined in `dodo.py`. At the moment the defaults are `pull_crsp_data` and `publish_docs`, so a bare `doit` handles both the data pipeline and the documentation build.

## Pulling CRSP Data
```bash
doit pull_crsp_data
```
What happens:
1. `src/pull_CRSP_Compustat.py` runs first (via `dodo.py`) and writes the WRDS pulls into `_data/`.
2. `src/build_crsp_data.py` loads those parquet files and produces a tidy excerpt.
3. The excerpt artifacts land at:
   - `_data/crsp_streamlit_excerpt.csv`
   - `_data/crsp_streamlit_excerpt.parquet`
   - `_data/crsp_data_metadata.json`
4. `app_04_crsp.py` reads the CSV excerpt, so relaunch the app after rerunning `doit` to see fresh data.

## Building and Publishing the Docs
```bash
doit publish_docs
```
Steps performed:
1. Sphinx builds the site to `_docs/build/html/`.
2. The HTML output is mirrored into the top-level `docs/` directory (old files are removed first, `.gitignore` is preserved).
3. GitHub Pages or any static host can serve from `docs/` without additional tweaks.

## Helpful Patterns
- `doit -n 1 <task>` runs just that task and prints live output.
- `doit clean` removes generated targets. Follow up with `doit` to rebuild from scratch.
- `doit forget pull_crsp_data` forces the CRSP pull to run again even if the targets already exist.

By keeping these workflows in `dodo.py`, you have a single entry point for both data refreshes and documentation builds—ideal when you’re iterating quickly during the workshop.
