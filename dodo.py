"""Project automation tasks using doit.

The focus is preparing CRSP data for the Streamlit demos. Running
`doit` (or `doit pull_crsp_data`) will attempt to fetch the CRSP stock
file via WRDS; if the connection fails, a synthetic sample is generated so
that the workshop can proceed offline. The task also writes a tidy excerpt
for the Streamlit app at `src/streamlit_examples/app_04_crsp.py`.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from settings import config

DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

DOCS_SRC_DIR = ROOT / "docs_src"
SPHINX_BUILD_ROOT = ROOT / "_docs"
SPHINX_HTML_DIR = SPHINX_BUILD_ROOT / "build" / "html"
DOCS_DIR = ROOT / "docs"


def task_pull_crsp_data():
    """Run the CRSP pull / synthesis pipeline and build the Streamlit excerpt."""

    return {
        "actions": [
            "python ./src/pull_CRSP_Compustat.py",
        ],
        "file_dep": [
            "./src/pull_CRSP_Compustat.py",
            "./src/config.py",
        ],
        "targets": [
            DATA_DIR / "CRSP_stock_ciz.parquet",
            DATA_DIR / "CRSP_Comp_Link_Table.parquet",
            DATA_DIR / "FF_FACTORS.parquet",
            DATA_DIR / "Compustat.parquet",
        ],
        "clean": True,
    }


def task_create_crsp_excerpt():
    """Create a CRSP excerpt for the Streamlit app."""
    return {
        "actions": ["python ./src/build_crsp_data.py"],
        "file_dep": [
            DATA_DIR / "CRSP_stock_ciz.parquet",
            DATA_DIR / "Compustat.parquet",
            DATA_DIR / "CRSP_Comp_Link_Table.parquet",
            DATA_DIR / "FF_FACTORS.parquet",
        ],
        "targets": [DATA_DIR / "crsp_streamlit_excerpt.csv"],
    }


def _iter_docs_dependencies() -> list[str]:
    return sorted(str(path) for path in DOCS_SRC_DIR.rglob("*") if path.is_file())


def task_build_docs():
    """Build the Sphinx site into the local _docs directory."""

    def _ensure_build_dir() -> None:
        (SPHINX_HTML_DIR).mkdir(parents=True, exist_ok=True)

    index_target = SPHINX_HTML_DIR / "index.html"

    return {
        "actions": [
            _ensure_build_dir,
            f"sphinx-build -b html {str(DOCS_SRC_DIR)} {str(SPHINX_HTML_DIR)}",
        ],
        "file_dep": _iter_docs_dependencies(),
        "targets": [index_target],
        "clean": True,
    }


def _copy_html_to_docs() -> None:
    if not SPHINX_HTML_DIR.exists():
        raise FileNotFoundError(
            "Sphinx HTML directory not found. Run `doit build_docs` first."
        )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for item in DOCS_DIR.iterdir():
        if item.name == ".gitignore":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in SPHINX_HTML_DIR.iterdir():
        destination = DOCS_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def task_publish_docs():
    """Copy built documentation into the `docs/` directory for publishing."""

    return {
        "actions": [_copy_html_to_docs],
        "file_dep": [str(SPHINX_HTML_DIR / "index.html"), *_iter_docs_dependencies()],
        "task_dep": ["build_docs"],
        "targets": [str(DOCS_DIR / "index.html")],
        "clean": True,
    }
