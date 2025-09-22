# Streamlit App Progression

## Core Streamlit Concepts

For comprehensive documentation on Streamlit fundamentals, refer to the official walkthrough:
**https://docs.streamlit.io/get-started/fundamentals/main-concepts**

### Essential Basics

**Running Apps:** Execute any Python script with `streamlit run your_script.py`. The app auto-reloads when you save changes.

**Data Flow:** Streamlit reruns your entire script top-to-bottom on every interaction. This makes apps behave like normal Python scripts.

**Display Methods:**
- **Magic commands**: Variables on their own line auto-render
- **`st.write()`**: Universal method for text, dataframes, charts
- **`st.dataframe()`**: Interactive tables with sorting/filtering
- **`st.line_chart()`, `st.bar_chart()`**: Quick built-in charts

**Widgets as Variables:**
```python
x = st.slider('Select value')  # Returns current slider value
option = st.selectbox('Choose', ['A', 'B', 'C'])  # Returns selected option
```

**Layout:**
- **`st.sidebar`**: Pin controls to the left panel
- **`st.columns`**: Create side-by-side layouts
- **`st.expander`**: Collapsible sections for space management

**Performance:** Use `@st.cache_data` to prevent expensive computations from rerunning on every interaction.

For deeper understanding of these concepts and advanced features, visit the official documentation linked above.

## Example Apps

We use four small apps to highlight how quickly Streamlit scales from a single chart to a data-rich dashboard. Run them in order to see how each layer adds new concepts. By now you should have executed `doit pull_crsp_data`, so the `_data/crsp_streamlit_excerpt.csv` file exists for the later examples.

## Environment
```bash
pip install -r requirements.txt
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

## App 04 – CRSP Snapshot (`src/streamlit_examples/app_04_crsp.py`)
- Mirrors the tear-sheet layout but reads the excerpt created by `doit pull_crsp_data` (`_data/crsp_streamlit_excerpt.csv`).
- Ideal for demonstrating how WRDS pulls flow through the pipeline: rerun `doit` and refresh the app to see new tickers or time ranges.
- Try customizing: surface additional CRSP attributes (e.g., shares outstanding, market cap rankings) or add narrative callouts describing the trends you observe.

## Tips for Iteration
- Use the sidebar to surface configuration switches instead of scattering inputs throughout the layout.
- Cache aggressively (`st.cache_data`) when loading large CSV/parquet files so Streamlit isn’t re-reading data with every interaction.
- Keep charts and metrics backed by small helper functions—this makes it easier to port the logic into your final project.
- When you’re ready to deploy, ensure `requirements.txt` contains every package you import and that you use relative paths (or environment variables) for data.

These scaffolds are intentionally lightweight; fork them, experiment, and mix/match components as you plan your own dashboard.
