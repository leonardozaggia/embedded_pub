"""
Shared loader for the translated Bayley-III structure (Objective 1).

All three ``b3_*`` concordance scripts compare the bottom-up clusters
(Objective 3) with the *final* translated structure, i.e. the 91-item taxonomy
in ``results/tables/bayley3_item_domain_assignments.csv`` (cascade pairing +
centroid assignment + expert review; built by ``analysis/item_assignments.py``).
Reading it from one place guarantees that Figure 5B, Figure S4, Figure S7 and
the concordance metrics all use the same item-to-domain map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

ASSIGNMENTS_REL = Path("results/tables/bayley3_item_domain_assignments.csv")

_LONG = {
    "ATT": "attention",
    "WM": "working_memory",
    "GDPS": "goal_directed_problem_solving",
    "FS": "flexibility_shift",
    "HOP": "higher_order_processing",
}


def assignments_path(root: Path) -> Path:
    p = root / ASSIGNMENTS_REL
    if not p.exists():
        raise FileNotFoundError(
            f"{ASSIGNMENTS_REL} not found -- run `python analysis/item_assignments.py` first")
    return p


def load_theoretical_dict(root: Path) -> Dict[str, List[str]]:
    """{factor_name (snake_case): [item_id, ...]} for the translated structure."""
    df = pd.read_csv(assignments_path(root))
    theo: Dict[str, List[str]] = {}
    for r in df.sort_values("item_number").itertuples():
        theo.setdefault(_LONG[r.domain], []).append(r.item_id)
    return theo
