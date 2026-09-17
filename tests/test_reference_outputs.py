"""
Regression tests: regenerated outputs must equal the frozen reference outputs.

Each test compares one live output file with its snapshot in tests/fixtures/
(see tests/fixtures/README.md).  Tests are skipped, not failed, when the live
file does not exist -- e.g. on a fresh clone without the restricted data --
so the suite stays runnable in CI while still catching silent drift after a
pipeline change on a machine that has the data.

Numeric CSV columns are compared exactly (the pipelines are deterministic:
fixed random seeds, committed embeddings, committed Mplus outputs); the JSON
metrics are compared to 4 decimals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

CSV_PAIRS = [
    ("reference_step1_cascade_pairs.csv", "data/outputs/step1_translation/b4_b3_cascade_pairs.csv"),
    ("reference_step2_cluster_assignments.csv",
     "data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr/03_clustering/cluster_assignments.csv"),
    ("reference_step3_cluster_assignments.csv",
     "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr/03_clustering/cluster_assignments.csv"),
    ("reference_item_domain_assignments.csv", "results/tables/bayley3_item_domain_assignments.csv"),
    ("reference_mismatches_ledger_b3.csv", "results/tables/mismatches_ledger_b3.csv"),
    ("reference_mismatches_ledger_b4.csv", "results/tables/mismatches_ledger_b4.csv"),
    ("reference_factor_item_map.csv", "psychometrics/factor_item_map.csv"),
    ("reference_difftest_results.csv", "psychometrics/mimic/difftest_results.csv"),
    ("reference_mimic_std_effects.csv", "psychometrics/mimic/mimic_std_effects.csv"),
    ("reference_outcomes_associations.csv", "psychometrics/outcomes/outcomes_associations.csv"),
    # Supplementary Tables S6-S8 (tables/make_supplementary_tables.py).  S8 can only be
    # rebuilt with the git-ignored review workbooks, so the committed CSVs are frozen here.
    ("reference_table_s6_b4_domain_recovery.csv", "results/tables/table_s6_b4_domain_recovery.csv"),
    ("reference_table_s7_b4_pair_level_intermediacy.csv", "results/tables/table_s7_b4_pair_level_intermediacy.csv"),
    ("reference_table_s8a_expert_review_b4.csv", "results/tables/table_s8a_expert_review_b4.csv"),
    ("reference_table_s8b_expert_review_b3.csv", "results/tables/table_s8b_expert_review_b3.csv"),
    # Table 1 (analysis/table1_baseline.py; level B, restricted data)
    ("reference_table1_baseline.csv", "results/tables/table1_baseline.csv"),
]

JSON_PAIRS = [
    ("reference_step2_comparison_metrics.json",
     "data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr/05_model_comparison/comparison_metrics.json"),
    ("reference_manuscript_metrics.json", "results/metrics/manuscript_metrics.json"),
    ("reference_sample_descriptives.json", "results/metrics/sample_descriptives.json"),
    ("reference_cohort_counts.json", "results/metrics/cohort_counts.json"),
    ("reference_expert_review_b3_kappa.json", "expert_review/records/coauthor_review_b3_kappa.json"),
    ("reference_expert_review_b4_kappa.json", "expert_review/records/coauthor_review_b4_kappa.json"),
]


def _live(rel: str) -> Path:
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not present (pipeline not run on this machine)")
    return p


def _round(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items() if k not in ("input", "date")}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


@pytest.mark.parametrize("fixture,live", CSV_PAIRS, ids=[f for f, _ in CSV_PAIRS])
def test_csv_matches_reference(fixture: str, live: str) -> None:
    ref = pd.read_csv(FIX / fixture)
    cur = pd.read_csv(_live(live))
    assert list(cur.columns) == list(ref.columns), "column layout changed"
    pd.testing.assert_frame_equal(cur.reset_index(drop=True), ref.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-9)


@pytest.mark.parametrize("fixture,live", JSON_PAIRS, ids=[f for f, _ in JSON_PAIRS])
def test_json_matches_reference(fixture: str, live: str) -> None:
    ref = json.loads((FIX / fixture).read_text(encoding="utf-8"))
    cur = json.loads(_live(live).read_text(encoding="utf-8"))
    assert _round(cur) == _round(ref)


def test_step1_model_spec_matches_reference() -> None:
    ref = yaml.safe_load((FIX / "reference_step1_model_spec.yaml").read_text(encoding="utf-8"))
    cur = yaml.safe_load(_live("data/outputs/step1_translation/mixed_centroids/b3_model_spec.yaml").read_text(encoding="utf-8"))
    (kr, vr), = ref.items()
    (kc, vc), = cur.items()
    assert kr == kc
    for key in ("factor_items", "factor_names", "factor_indices", "n_items", "n_factors"):
        assert vc.get(key) == vr.get(key), f"{key} differs from the frozen step-1 specification"


@pytest.mark.parametrize("name", ["model_cfa_3factor.out", "model_unidimensional_difftest.out"])
def test_mplus_output_matches_reference(name: str) -> None:
    ref = (FIX / f"reference_{name}").read_bytes()
    cur = _live(f"psychometrics/measurement/{name}").read_bytes()
    assert cur.replace(b"\r\n", b"\n") == ref.replace(b"\r\n", b"\n")
