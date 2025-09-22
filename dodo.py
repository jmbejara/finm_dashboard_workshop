"""Project automation tasks using doit.

The pipeline pulls CRSP data (live via WRDS when available, synthetic otherwise),
prepares a Streamlit-friendly excerpt, and rebuilds the Sphinx documentation.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable
import subprocess

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from settings import config

# Paths & artefacts ---------------------------------------------------------
DATA_DIR: Path = config("DATA_DIR")
DOCS_SRC_DIR = ROOT / "docs_src"
DOCS_BUILD_DIR: Path = config("DOCS_BUILD_DIR")
SPHINX_HTML_DIR = DOCS_BUILD_DIR / "build" / "html"
DOCS_DIR = ROOT / "docs"

CRSP_PARQUET = DATA_DIR / "CRSP_stock_ciz.parquet"
CRSP_COMP_LINK = DATA_DIR / "CRSP_Comp_Link_Table.parquet"
CRSP_FACTORS = DATA_DIR / "FF_FACTORS.parquet"
CRSP_COMPUSTAT = DATA_DIR / "Compustat.parquet"

DERIVED_DIR = DATA_DIR / "derived"
EXCERPT_CSV = DERIVED_DIR / "crsp_streamlit_excerpt.csv"
EXCERPT_PARQUET = DERIVED_DIR / "crsp_streamlit_excerpt.parquet"
EXCERPT_METADATA = DERIVED_DIR / "crsp_data_metadata.json"


DOIT_CONFIG = {
    "default_tasks": [
        "pull_crsp_data",
        "create_crsp_excerpt",
        "publish_docs",
    ],
}


# CRSP tasks -----------------------------------------------------------------


def _pull_crsp_with_fallback() -> None:
    """Try the WRDS pull; fall back to synthetic data if it fails."""

    try:
        subprocess.run(
            ["python", "./src/pull_CRSP_Compustat.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"WRDS pull failed ({exc}); generating synthetic data instead.")
        subprocess.run(
            ["python", "./src/build_crsp_data.py"],
            check=True,
        )


def task_pull_crsp_data():
    """Pull CRSP/Compustat/FF data via WRDS (with synthetic fallback)."""

    return {
        "actions": [_pull_crsp_with_fallback],
        "file_dep": [
            "./src/pull_CRSP_Compustat.py",
            "./src/settings.py",
        ],
        "targets": [CRSP_PARQUET, CRSP_COMP_LINK, CRSP_FACTORS, CRSP_COMPUSTAT],
        "clean": True,
    }


def task_create_crsp_excerpt():
    """Create the Streamlit excerpt based on the CRSP pulls."""

    return {
        "actions": [["python", "./src/build_crsp_data.py"]],
        "file_dep": [
            CRSP_PARQUET,
            "./src/build_crsp_data.py",
            "./src/settings.py",
        ],
        "targets": [EXCERPT_CSV, EXCERPT_PARQUET, EXCERPT_METADATA],
        "task_dep": ["pull_crsp_data"],
        "clean": True,
    }


def task_show_crsp_excerpt_info():
    """Print a quick summary once the excerpt exists."""

    def _print_summary() -> None:
        if not EXCERPT_CSV.exists():
            print("CRSP excerpt not found. Run `doit create_crsp_excerpt` first.")
            return
        size = EXCERPT_CSV.stat().st_size
        print(
            f"CRSP excerpt ready at {EXCERPT_CSV} (size={size} bytes). "
            "Launch `streamlit run src/streamlit_examples/app_04_crsp.py` to explore it."
        )

    return {
        "actions": [_print_summary],
        "verbosity": 2,
        "task_dep": ["create_crsp_excerpt"],
    }


# Documentation tasks -------------------------------------------------------


def _iter_docs_dependencies() -> Iterable[str]:
    return sorted(str(path) for path in DOCS_SRC_DIR.rglob("*") if path.is_file())


def task_build_docs():
    """Build the Sphinx site into `_docs/build/html`."""

    def _ensure_build_dir() -> None:
        SPHINX_HTML_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "actions": [
            _ensure_build_dir,
            [
                "sphinx-build",
                "-b",
                "html",
                str(DOCS_SRC_DIR),
                str(SPHINX_HTML_DIR),
            ],
        ],
        "file_dep": list(_iter_docs_dependencies()),
        "targets": [SPHINX_HTML_DIR / "index.html"],
        "clean": True,
    }


def _copy_html_to_docs() -> None:
    if not SPHINX_HTML_DIR.exists():
        raise FileNotFoundError("Sphinx output missing. Run `doit build_docs` first.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for item in DOCS_DIR.iterdir():
        if item.name == ".gitignore":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in SPHINX_HTML_DIR.iterdir():
        target = DOCS_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def task_publish_docs():
    """Copy the built HTML into `docs/` for publishing."""

    return {
        "actions": [_copy_html_to_docs],
        "file_dep": [SPHINX_HTML_DIR / "index.html", *list(_iter_docs_dependencies())],
        "task_dep": ["build_docs"],
        "targets": [DOCS_DIR / "index.html"],
        "clean": True,
    }
