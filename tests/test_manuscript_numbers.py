"""
Every statistic reported in the manuscript must be reproduced by the outputs.

Thin pytest wrapper around analysis/verify_manuscript_numbers.py (which holds
the expected values transcribed from the paper).  Skipped when the regenerated
outputs are not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

REQUIRED = [
    "results/metrics/sample_descriptives.json",
    "results/metrics/manuscript_metrics.json",
    "results/tables/bayley3_item_domain_assignments.csv",
    "results/tables/table_s4a_mimic_covariate_associations.csv",
    "results/tables/table1_baseline.csv",
    "results/metrics/cohort_counts.json",
]


def _checks():
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            pytest.skip(f"{rel} missing -- run the pipeline first (docs/REPRODUCING.md)")
    import verify_manuscript_numbers as v
    return v.run()


def test_all_manuscript_numbers_reproduced() -> None:
    checks = _checks()
    failed = [c for c in checks if not c["ok"]]
    msg = "\n".join(f"{c['section']}: {c['label']} expected {c['expected']} observed {c['observed']}" for c in failed)
    assert not failed, f"{len(failed)} manuscript number(s) not reproduced:\n{msg}"
