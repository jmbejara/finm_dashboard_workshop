# Quick Streamlit Inspiration (`app_01.py`)

Before we dive into tooling and data pulls, launch the full tear-sheet demo to see what you’ll build by the end of the workshop.

## Run the App
```bash
pip install -r requirements.txt
streamlit run src/streamlit_examples/app_01.py
```

What you’ll see:
- Multi-ticker comparison with cumulative performance, summary metrics, and a naïve forecast preview.
- Automatic fallbacks to bundled sample data (or synthetic prices) when you don’t have network access. That’s expected at this stage.
- A structure we’ll reuse when we wire in WRDS/CRSP data from `doit` and forecasts from FTSFR.

Take a minute to click through the tabs and sidebar controls. Note any layout features or narrative touches you’d like to keep when you customize your own dashboard.

Once you’ve previewed the app, return to the main discussion outline—we’ll set up the data pipeline next.
