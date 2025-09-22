# Workshop Introduction

Welcome to **From Data to Dashboard: Building Interactive Financial Visualizations in Python**. I’m Jeremy Bejarano. I works with the Office of Financial Research and I teach in the University of Chicago’s Financial Mathematics program. My Ph.D. is in economics from the University of Chicago, but I specialize in asset pricing. At the Office of Financial Research, I work on topics in financial stability and specialize in short-term funding markets (especially repo markets). In the Financial Mathematics program, I teach a course called ["FINM 32900: Full Stack Quantitative Finance"](https://finmath.uchicago.edu/curriculum/degree-concentrations/financial-computing/finm-32900/). I’ll be guiding you through two hands-on days focused on translating WRDS data and forecasting results into polished Streamlit dashboards.

## What You’ll Accomplish
- Run through a progressive series of Streamlit apps so you can iterate from “hello world” to a fully fledged financial tear sheet.
- Use our automated `doit` pipeline to pull CRSP data (or generate an offline-friendly substitute) and publish workshop documentation with Sphinx.
- Clone the Financial Time-Series Forecasting Repository (FTSFR) and learn how its datasets and forecasting utilities plug into your dashboard.
- Leave with a working Streamlit project, a playbook for sourcing WRDS data, and a roadmap for incorporating forecasting models.

## Who This Workshop Is For
- Graduate students and analysts with basic Python proficiency who want to sharpen their communication and visualization skills.
- Participants who either have WRDS access already or plan to work with the synthetic data we provide while credentials are being finalized.
- Anyone curious about building interactive narratives around financial datasets without over-engineering the stack.

## How the Two Days Are Structured
1. **Day 1 – Foundations:** tour the Streamlit apps, get comfortable with the tooling (`pip`, `doit`, `sphinx-build`), and make your first customizations using CRSP-style data.
2. **Day 2 – Benchmarking & Planning:** deep dive on the FTSFR benchmark, align on forecasting concepts, and plan how you will integrate forecasts into your dashboards ahead of the final presentations.

Throughout both sessions we’ll alternate short lectures with guided labs and open build time. Bring a laptop, a WRDS account if you have one, and a short list of analyses you’re excited to visualize.

## Staying in Touch
Questions or blockers? Reach out by email (jbejarano@uchicago.edu). I’ll provide help during the breakout blocks and can schedule 1:1 time if you need extra debugging help.

Let’s build something you’ll be proud to demo. When you’ve read through this introduction, move to the [quick Streamlit example](quick_streamlit_example.md) and launch `app_01.py` for an inspirational preview.
