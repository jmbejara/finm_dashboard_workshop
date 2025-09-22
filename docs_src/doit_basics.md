# Automation with `doit`

This repository uses [`doit`](https://pydoit.org/) to manage data pulls, documentation builds, and other repeatable tasks. Think of it as a Python-friendly build system that handles dependencies automatically.

## Installation
```bash
pip install -r requirements.txt
```
This installs `doit` alongside the other packages used in the workshop.

## Primary Usage: Run Everything
```bash
doit
```
This single command runs all tasks in the proper order based on their dependencies and targets. `doit` automatically:
- Determines which tasks need to run based on file timestamps and dependencies
- Executes tasks in the correct order
- Skips tasks whose targets are already up-to-date
- Handles the entire pipeline: data pulls → processing → documentation build

### Parallel Execution
Speed up the build with parallel task execution:
```bash
doit -n 4  # Run up to 4 tasks in parallel
```
This is especially useful when you have multiple independent data processing or documentation tasks.

## What Gets Built
Running `doit` executes the default tasks defined in `dodo.py`:
1. **Data Pipeline**: Pulls CRSP data from WRDS (or generates synthetic data as fallback), then creates excerpts
2. **Documentation**: Builds and publishes the Sphinx documentation

Output artifacts:
- `_data/crsp_streamlit_excerpt.csv` – Data for Streamlit apps
- `_data/crsp_streamlit_excerpt.parquet` – Efficient data format
- `_data/crsp_data_metadata.json` – Data pipeline metadata
- `docs/` – Published documentation site

## Rebuilding from Scratch
```bash
doit clean  # Remove all generated files
doit        # Rebuild everything
```

## Advanced Usage: Individual Tasks

While `doit` handles everything automatically, you can inspect or run individual tasks when needed:

```bash
doit list  # Show all available tasks
```

Key tasks include:
- `pull_crsp_data` – WRDS data pull and processing
- `show_crsp_excerpt_info` – Display data summary
- `build_docs` – Generate Sphinx documentation
- `publish_docs` – Copy docs for GitHub Pages

Run a specific task:
```bash
doit pull_crsp_data  # Run just the CRSP data pipeline
```

Force a task to rerun:
```bash
doit forget pull_crsp_data  # Clear task status
doit pull_crsp_data         # Run again
```

## Tips
- After running `doit`, restart your Streamlit apps to see fresh data
- Use `doit -n 1` for verbose output during debugging
- The `dodo.py` file defines all task dependencies and build rules

By using `doit` as your primary command, you ensure all components stay synchronized—ideal for rapid iteration during the workshop.
