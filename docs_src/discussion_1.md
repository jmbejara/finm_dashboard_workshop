# Discussion 1 – Kickoff & Dashboard Preview

Day 1 is all about orienting ourselves, exploring the tooling, and leaving with everything installed so you can iterate overnight.

- Start with the [workshop introduction](introduction.md) to revisit goals, prerequisites, and how the two days fit together.
- Walk through the [Streamlit basics](streamlit_basics.md) document while we demo the four scaffold apps.
- Follow the [doit basics](doit_basics.md) guide to run the automation pipeline (`doit pull_crsp_data` → publish docs).
- Use [Intro to FTSFR data](intro_to_ftsfr_data.md) as a reference once you’re ready to clone the external repository.

## Session Goals
- Understand the workshop deliverable and how we’ll collaborate over the next two days.
- Launch and customize the Streamlit examples (`app_02`, `app_03`, `app_01`, `app_04_crsp`).
- Run the `doit` pipeline to produce a CRSP excerpt and rebuild the documentation site.
- Confirm you can clone the FTSFR repo and pull WRDS data (or capture questions if credentials are pending).

## Agenda — Day 1

### Segment A · Lecture & Live Demo (12:00 – 1:30 pm)
1. Orientation: revisit goals, grading rubric, and deployment expectations.
2. Progressive Streamlit tour (apps 02 → 03 → 01 → 04) with code comparisons.
3. Discussion: what makes a financial dashboard credible? Capture criteria the group agrees on.

### Segment B · Breakout Lab (1:30 – 2:30 pm)
- Customize the hello world app, point the intermediate app to a new CSV, and run `doit pull_crsp_data` so `app_04_crsp.py` reflects your own data slice.
- Optional stretch: attempt a Streamlit Community Cloud deployment of your favorite variant.

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
