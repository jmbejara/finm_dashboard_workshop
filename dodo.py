"""Project automation tasks using doit.

The pipeline pulls CRSP data (live via WRDS when available, synthetic otherwise),
prepares a Streamlit-friendly excerpt, and rebuilds the Sphinx documentation.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from settings import config

DATA_DIR: Path = config("DATA_DIR")
DOCS_SRC_DIR = ROOT / "docs_src"
DOCS_BUILD_DIR: Path = config("DOCS_BUILD_DIR")
SPHINX_HTML_DIR = DOCS_BUILD_DIR / "build" / "html"
DOCS_DIR = ROOT / "docs"


def task_pull_crsp_data():
    """Pull CRSP/Compustat/FF data via WRDS (with synthetic fallback)."""

    return {
        "actions": [
            "python ./src/pull_CRSP_Compustat.py",
        ],
        "file_dep": [
            "./src/pull_CRSP_Compustat.py",
            "./src/settings.py",
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
    """Create the Streamlit excerpt based on the CRSP pulls."""

    return {
        "actions": [
            "python ./src/build_crsp_data.py",
        ],
        "file_dep": [
            DATA_DIR / "CRSP_stock_ciz.parquet",
            "./src/build_crsp_data.py",
            "./src/settings.py",
        ],
        "targets": [
            DATA_DIR / "crsp_streamlit_excerpt.csv",
            DATA_DIR / "crsp_streamlit_excerpt.parquet",
            DATA_DIR / "crsp_data_metadata.json",
        ],
        "task_dep": ["pull_crsp_data"],
        "clean": True,
    }


def task_pull_yfinance():
    """Download benchmark yfinance data for the Streamlit fallback."""

    return {
        "actions": [
            "python ./src/pull_yfinance.py",
        ],
        "file_dep": [
            "./src/pull_yfinance.py",
            "./src/settings.py",
        ],
        "targets": [
            DATA_DIR / "yfinance_prices.parquet",
            DATA_DIR / "yfinance_prices.csv",
        ],
        "clean": True,
    }


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

    nojekyll = DOCS_DIR / ".nojekyll"
    nojekyll.write_text("\n")


def task_publish_docs():
    """Copy the built HTML into `docs/` for publishing."""

    return {
        "actions": [_copy_html_to_docs],
        "file_dep": [SPHINX_HTML_DIR / "index.html", *list(_iter_docs_dependencies())],
        "task_dep": ["build_docs"],
        "targets": [DOCS_DIR / "index.html"],
        "clean": True,
    }
