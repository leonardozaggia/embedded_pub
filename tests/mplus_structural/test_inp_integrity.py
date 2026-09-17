"""
Structural integrity checks for every tracked Mplus .inp file.

Two properties every .inp file must have (see also psychometrics/checks/):
no UTF-8 BOM, which makes Mplus reject the file with a generic,
line-number-free error, and a `NAMES ARE` variable list whose length equals
the column count of combined.dat, since a shorter or longer list makes Mplus
silently read every column after the divergence as the wrong variable.

Neither check requires Mplus itself -- they're pure text/structural
validation, so they run in CI without a license.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PSYCHOMETRICS_DIR = REPO_ROOT / "psychometrics"

UTF8_BOM = b"\xef\xbb\xbf"


def _all_inp_files() -> list[Path]:
    return sorted(PSYCHOMETRICS_DIR.rglob("*.inp"))


def _combined_dat_width() -> int:
    with open(PSYCHOMETRICS_DIR / "varnames.txt") as f:
        return len([line for line in f if line.strip()])


@pytest.mark.parametrize("inp_path", _all_inp_files(), ids=lambda p: str(p.relative_to(PSYCHOMETRICS_DIR)))
def test_inp_file_has_no_bom(inp_path: Path) -> None:
    with open(inp_path, "rb") as f:
        head = f.read(3)
    assert head != UTF8_BOM, (
        f"{inp_path} starts with a UTF-8 BOM -- Mplus 8.6 fails to parse this "
        f"file with a generic 'does not contain valid commands' error. Strip "
        f"the first 3 bytes (content-preserving fix)."
    )


@pytest.mark.parametrize("inp_path", _all_inp_files(), ids=lambda p: str(p.relative_to(PSYCHOMETRICS_DIR)))
def test_inp_names_are_matches_combined_dat_width(inp_path: Path) -> None:
    text = inp_path.read_text(encoding="utf-8", errors="replace")
    # Only .inp files that actually read combined.dat are in scope; some
    # supplementary/example .inp files may use a different data source.
    if "combined.dat" not in text:
        pytest.skip(f"{inp_path.name} does not reference combined.dat")

    match = re.search(r"NAMES\s+ARE\s+(.*?);", text, re.IGNORECASE | re.DOTALL)
    assert match is not None, f"{inp_path} references combined.dat but has no NAMES ARE statement"

    names = match.group(1).split()
    expected_width = _combined_dat_width()
    assert len(names) == expected_width, (
        f"{inp_path}: NAMES ARE lists {len(names)} variables but "
        f"combined.dat/varnames.txt has {expected_width} columns -- this "
        f"silently misreads every column after the shorter list ends."
    )


def test_varnames_matches_combined_dat_column_count() -> None:
    import numpy as np

    dat_path = PSYCHOMETRICS_DIR / "combined.dat"
    if not dat_path.exists():
        pytest.skip("combined.dat not present (requires data/raw/ + prepare_data.R)")

    n_cols = np.loadtxt(dat_path, dtype=float).shape[1]
    assert n_cols == _combined_dat_width()
