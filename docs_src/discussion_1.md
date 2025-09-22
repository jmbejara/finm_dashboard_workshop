# Discussion 1 – Kickoff & Dashboard Preview

Day 1 is all about getting you access to the data and getting you up and running with a few example codes. This will allow you to iterate overnight and be ready for the next day.

- Start with the [workshop introduction](introduction.md) to revisit goals, prerequisites, and how the two days fit together.
- Walk through the [Streamlit basics](streamlit_basics.md) document while we demo the four scaffold apps.
- Follow the [doit basics](doit_basics.md) guide to run the automation pipeline (`doit pull_crsp_data` → publish docs).
- Use [Intro to FTSFR data](intro_to_ftsfr_data.md) as a reference once you’re ready to clone the external repository.

## Session Goals
- Understand the workshop deliverable and how we’ll collaborate over the next two days.
- Launch and customize the Streamlit examples (`app_01`, `app_02`, `app_03`, `app_04_crsp`).
- Run the `doit` pipeline to produce a CRSP excerpt and rebuild the documentation site.
- Confirm you can clone the FTSFR repo and pull WRDS data (or capture questions if credentials are pending).

## Agenda — Day 1

### Segment A · Lecture & Live Demo (12:00 – 1:30 pm)
1. Orientation: Who am I? What are the goals of the workshop?
2. Progressive Streamlit tour with code comparisons.
3. Discussion: What makes a financial dashboard useful? 
4. Preview the Financial Time-Series Forecasting Repository (FTSFR) and how it fits into the workshop. What other asset classes might you be interested in?

### Segment B · Breakout Lab (1:30 – 2:30 pm)
- Customize the hello world app, point the intermediate app to a new CSV (maybe from FTSFR)
- Develop your own visualizations of the data (e.g., from a different asset class)

### Segment C · Lecture & Guided Walkthrough (2:30 – 4:00 pm)
1. Deep dive on `doit` and the Sphinx publishing flow.
2. Clone the FTSFR repository, configure `.env` + `subscriptions.toml`, and test WRDS access.
3. Export a slim CRSP CSV for further visualization work.

### Independent Practice (4:00 – 5:00 pm)
- Finish personalizing the apps and document two dashboard ideas you want to explore in Discussion 2.

## Suggested Next Steps Before Discussion 2
- Spend time with each app (`app_02`, `app_03`, `app_04_crsp`, `app_01`) and note one enhancement you want to attempt tomorrow.
- Finish cloning the FTSFR repository, configure credentials, and pull the CRSP dataset (export a workshop-friendly CSV for quick iteration).
- Note two questions or ideas about forecasting/narrative framing that you want addressed in the benchmarking lecture.

```{toctree}
:maxdepth: 1
:hidden:
introduction.md
streamlit_basics.md
doit_basics.md
intro_to_ftsfr_data.md
```
