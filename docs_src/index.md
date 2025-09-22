# From Data to Dashboard: Building Interactive Financial Visualizations in Python

Turn your financial analyses into interactive web dashboards that look as polished as the insights they communicate. This two-day workshop gives students with foundational Python experience the structure, support, and hands-on time they need to source data, prototype compelling visuals, and publish a shareable dashboard.

## Workshop Overview

- **Instructor:** Jeremy Bejarano
- **Email:** jbejarano@uchicago.edu
- **Audience:** Students comfortable with basic Python who want to level up their data storytelling and presentation skills.
- **Tools:** Python, WRDS for data access, Streamlit for interactive dashboards, plus supporting open-source visualization libraries.
- **Format:** Each day alternates focused lectures with guided breakout sessions so you can immediately apply what you learn. Expect plenty of time for questions and in-the-room troubleshooting.

## Required Software

Make sure the following tools are installed and ready before Day 1:

- **Python 3.11+ (Anaconda recommended)** – Install the [Anaconda distribution](https://www.anaconda.com/products/distribution) and verify `streamlit hello` runs in your environment.
- **VS Code (or another editor)** – Grab [VS Code](https://code.visualstudio.com/) plus the Python and Jupyter extensions, or bring your preferred IDE.
- **Git + GitHub account** – Install [Git](https://git-scm.com/downloads) and set up a GitHub account if you don’t already have one.
- **WRDS credentials** – Request an account and test your connection ahead of time:
  ```python
  import wrds
  db = wrds.Connection(wrds_username="your_username")
  ```
  WRDS may prompt you to create a `.pgpass` file; select `Y` if asked.

## What You'll Learn

- Pulling and wrangling financial datasets from the Wharton Research Data Services (WRDS) platform.
- Designing visual narratives that highlight the story behind your analysis.
- Building interactive dashboards in Streamlit that you can run locally or publish to the web.
- Packaging your work so that stakeholders can navigate, explore, and trust your results.

## Detailed Schedule

### Monday, September 22
- Discussion 1 - 12 - 1:30 pm : Kickoff, Streamlit tour, tooling setup
- Breakout Session - 1:30 - 2:30 pm
- Discussion 2 – 2:30 - 4:00 pm: Benchmark slides and forecasting repository overview

### Tuesday, September 23
- Discussion 3 – 12:00 - 1:30 pm: Forecast integration and dashboard polish
- Breakout Session - 1:30 - 2:30 pm
- Discussion 4 – 2:30 - 4:00 pm: Final presentations and retrospectives

## Project Focus

Across both days you will scope, build, and present a custom financial dashboard:

1. Identify a question or analysis that matters to you or your team. You might choose to explore a specific asset class or other dataset available in [ftsfr](https://github.com/jmbejara/ftsfr).
2. Use WRDS to collect, clean, and validate the underlying data.
3. Prototype visualizations that surface the key findings.
4. Assemble the final interactive dashboard in Streamlit and publish it locally, with optional guidance for hosting on the web.

Expect plenty of instructor feedback during the breakout blocks, plus practical tips on layout, performance, and narrative flow. There will also be room to explore advanced touches—like scenario toggles or comparison views—for teams who move quickly.

## Preparation Checklist

- Confirm the required software and WRDS credentials are working.
- Bring a laptop, charger, and any prior analysis you might want to showcase.
- Consider a data question you are excited to explore; we will workshop ideas during the opening breakout.

## Helpful References

- [MyST Markdown Demo](myst_markdown_demos.md) — quick primer on formatting notebook-friendly notes for the workshop.
- [Syncing Files with Git and GitHub](syncing_files_with_git_and_github.md) — keep your project version-controlled as you iterate.
- [Using CRSP Data](using_CRSP_data.md) — background on a high-value WRDS dataset you might draw from.
- [WRDS Intro and Web Queries](WRDS_intro_and_web_queries.md) — step-by-step guide to pulling data for your dashboard.

```{toctree}
:maxdepth: 1
:caption: Workshop Sessions 📈
:hidden:
discussion_1.md
discussion_2.md
discussion_3.md
discussion_4.md
```

```{toctree}
:maxdepth: 1
:caption: Useful References 📖
:hidden:
myst_markdown_demos.md
syncing_files_with_git_and_github.md
using_CRSP_data.md
WRDS_intro_and_web_queries.md
```

## Stay in Touch

Questions, suggestions, or tweaks to the plan? Reach out anytime—Jeremy is refining the flow based on your feedback and is looking forward to building with you.
