# Discussion 3 – Forecast Integration & Narrative Design

**UNDER CONSTRUCTION**

## Session Goals
- Integrate forecast outputs into the Streamlit dashboard with clear explanations and uncertainty visuals.
- Pressure-test performance: caching strategies, lightweight data extracts, and user experience tweaks.
- Coach students on storytelling: how to frame investment takeaways, risk caveats, and next steps for stakeholders.
- Allocate build time so teams can approach a feature-complete dashboard before the final session.

## Kickoff (12:00 – 12:20 pm)
- Show-and-tell: two volunteers demo their forecast visual and receive rapid feedback.
- Quick poll on blockers (data, modeling, Streamlit, deployment); triage into mini-coaching clusters.

## Segment A · Mini-Lecture (12:20 – 1:00 pm)
- Communicating forecasts responsibly: prediction intervals, scenario framing, and disclosure of model limitations.
- Streamlit UX patterns for multipage dashboards, tabs vs. accordions, and layout containers for responsive design.
- Performance tips: caching with `st.cache_data`, minimizing recompute cost, and lazy-loading heavy CSVs.

## Segment B · Guided Build (1:00 – 1:40 pm)
- Convert forecast CSV output into a fan chart or ribbon plot (Plotly `go.Scatter` with confidence bands).
- Add user controls: horizon slider, model selector (baseline vs. advanced), and toggle for “compare to historical avg.”
- Wire in narrative callouts—`st.caption`, `st.warning`, or custom markdown—to explain what the chart shows and what users should look for.

## Segment C · Breakout Lab (1:40 – 2:40 pm)
- Goal: produce a draft “Forecast” tab in the dashboard that meets the rubric (visual + plain-language takeaway + metric table).
- Instructor roves to help debug data joins or layout glitches.
- Deliverable for the hour: push a branch update or save a ZIP backup; post a screenshot to the cohort chat.

## Segment D · Lecture & Discussion (3:00 – 3:40 pm)
- Benchmarking recap from FTSFR: highlight findings from `draft_ftsfr.tex` (e.g., challenges forecasting returns vs. spreads) and connect to student projects.
- Facilitate a conversation around model risk and ethical deployment (when not to ship a forecast, how to document caveats).

## Segment E · Open Build Time (3:40 – 4:00 pm)
- Students continue polishing dashboards; schedule 1:1 check-ins for early-day troubleshooting tomorrow.

## Exit Tickets & Prep for Discussion 4
- Submit a short update: remaining TODOs, deployment status, and who will present for the team.
- Prepare any data refresh scripts you’ll need to rerun before the final showcase.
