# Deploy to Streamlit Community Cloud

Launch your workshop app so classmates (and future stakeholders) can explore it from any browser. The Community Cloud handles hosting, requirements installation, and authentication—you just connect your GitHub repo and point to the Streamlit entry point.

## Prerequisites
- GitHub repository containing your Streamlit app and `requirements.txt`
- Clean `git status` (all changes committed and pushed)
- Conda environment not required on the cloud—Streamlit installs dependencies from `requirements.txt`

## 1. Link Streamlit Cloud to GitHub
1. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
2. Authorize Streamlit to access your repositories (organization owners may need to grant additional access).
3. You’ll land on the **App dashboard**, which lists any existing deployments.

## 2. Create a New App
1. Click **New app**.
2. Select the repository, branch (`main` by default), and the path to your Streamlit file (e.g., `app.py` or `src/streamlit_app.py`).
3. Optionally enter a custom app name and description.
4. Click **Deploy**—Streamlit queues a build, installs dependencies from `requirements.txt`, and launches the app.

## 3. Monitor the Build
- Watch the logs for missing dependencies or path issues.
- If installation fails, update `requirements.txt` locally, commit, push, and click **Rerun**.
- Confirm the app loads and interactive elements behave as expected before sharing the link.

## 4. Share Your Live App
- Copy the app URL (format: `https://<app-name>-<username>.streamlit.app`).
- Post the link in the cohort chat and add it to your README for easy access.
- Reference example: [https://finmdashboardworkshop.streamlit.app/](https://finmdashboardworkshop.streamlit.app/).

## Community vs. Pro
- **Community Cloud (Free):** public-only apps, limited resource allocations, manual secrets management via the web UI. Perfect for class demos and portfolio pieces.
- **Pro / Snowflake Streamlit:** private apps, SSO options, larger resource quotas, and team-level governance. Ideal for production use cases inside organizations.

## Tips & Common Fixes
- **Secrets:** click **⋮ → Edit secrets** to add API keys; access them in code via `st.secrets`.
- **Long-running jobs:** use caching (`st.cache_data`) or background tasks to keep the UI responsive.
- **Large data files:** store them in cloud storage (S3, Google Drive) or generate them on the fly—avoid uploading huge datasets to the repo.
- **App not updating?** Make sure you committed and pushed the latest changes; Streamlit rebuilds on every new commit to the selected branch.

Once the deployment succeeds, practice the demo flow so you’re ready for the showcase in Discussion 4.
