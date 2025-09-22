# Stand Up Your Workshop Repository

Get your Streamlit app under version control so you can collaborate, deploy, and iterate without fear. Follow this lab as a checklist—by the end, you’ll have a clean GitHub repository that contains only the code and assets required to run your app.

## Before You Begin
- Git and VS Code (or your editor of choice) installed locally
- GitHub account verified with 2FA
- Streamlit app files in a working directory on your laptop
- Conda environment activated (`conda activate finm`) so your terminal is ready for later steps

## 1. Create the Repository on GitHub
1. Visit [https://github.com/new](https://github.com/new).
2. Name the repo something descriptive (e.g., `finm-dashboard-<teamname>`).
3. Keep **Public** checked so Streamlit Community Cloud can access it.
4. Add a short description and initialize with a `README.md`. Skip adding a `.gitignore`—we’ll create it locally so you can customize exclusions.
5. Click **Create repository** and copy the repository URL (`https://github.com/<user>/<repo>.git` or SSH variant if you’ve configured keys).

## 2. Clone the Repo Locally
```bash
# Replace with your GitHub username and repository name
git clone https://github.com/<user>/<repo>.git
cd <repo>
```
If you prefer SSH and have already uploaded your public key to GitHub:
```bash
git clone git@github.com:<user>/<repo>.git
```
Tip: Run `git remote -v` to confirm the clone points to the correct origin.

## 3. Stage Your App Skeleton
Organize your Streamlit app inside the repo. A typical structure looks like:
```
.
├── README.md
├── requirements.txt
└── app.py  # or src/streamlit_app.py
```
Copy or move in only the files needed to run the app:
- Core Streamlit script(s)
- Lightweight data samples (small CSV/Parquet files) if the app requires them
- Supporting modules or utility scripts
- `requirements.txt` listing runtime dependencies
Avoid committing secrets, WRDS credentials, or large raw datasets—leave those out or reference them in the README.

## 4. Check the Working Tree
```bash
git status
```
You should see your new files listed as “untracked.” If anything unexpected appears (e.g., `__pycache__/`, `.DS_Store`), create a `.gitignore` now:
```bash
echo "__pycache__/\n.DS_Store\n_data/" >> .gitignore
```
Run `git status` again to ensure only the files you intend to share remain.

## 5. Inspect Changes with Diffs
Before staging, review what will be committed:
```bash
git diff -- app.py
```
Use `git diff --staged` later to double-check staged content. Encourage teammates to glance at diffs before every commit—this prevents surprises in your history.

## 6. Stage Intentionally
Add files explicitly rather than using `git add .`:
```bash
git add app.py requirements.txt README.md .gitignore
```
If you have subdirectories, list them by path (e.g., `git add src/streamlit_components.py`). Keeping the staging step intentional reinforces that only production-ready assets land in the repo.

## 7. Commit with Context
```bash
git commit -m "Add initial Streamlit app scaffold"
```
Commit messages should describe *why* the change matters. Examples:
- `Add CRSP return visualization to main app`
- `Document local setup steps in README`

If Git requests your name or email, configure them once:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@uchicago.edu"
```

## 8. Push to GitHub
```bash
git push -u origin main
```
The `-u` flag links your local `main` branch to the remote `main`. Future pushes can omit the upstream:
```bash
git push
```
If you see an authentication prompt, use a personal access token or SSH key—password login is no longer supported.

## 9. Verify the Remote
1. Refresh your GitHub repository page.
2. Confirm that the files render correctly and code previews look as expected.
3. Copy the repository URL—you’ll need it when you configure Streamlit Community Cloud.

## 10. Ongoing Workflow Tips
- Run `git status` early and often; aim for a “clean tree” before breaks.
- Use focused branches (`git checkout -b feature/add-forecast-tab`) when collaborating; open pull requests for review.
- Keep commits small and meaningful so you can revert or cherry-pick easily.
- Push after each major milestone (e.g., deployment success, new visualization) so you always have a safe restore point.

## Troubleshooting
- **SSH permission denied:** confirm your public key is loaded with `ssh-add -l`; if empty, run `ssh-add ~/.ssh/id_rsa` (or your key file).
- **HTTP authentication failed:** create a personal access token at [https://github.com/settings/tokens](https://github.com/settings/tokens) with `repo` scope and use it in place of a password.
- **Accidentally staged a large file:** run `git reset HEAD <file>` to unstage, remove it from disk or `.gitignore`, then stage again.
- **Merge conflicts:** ask a classmate to walk through `git status` output; resolve conflicts in your editor, then `git add` the fixed files and `git commit`.

Celebrate once your push lands—your deployment pipeline now has a reliable source of truth.
