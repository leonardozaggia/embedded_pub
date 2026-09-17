#!/usr/bin/env python
"""
Pair-level intermediacy of the Bayley-4 clusters (Discussion claim).

The Discussion states that the unsupervised clusters "matched each of
[Aylward's two reference models] more closely than they matched each other,
placing it between the two rather than outside both" (ARI 0.49 / 0.42 vs
0.19).  ARI alone cannot distinguish genuine intermediacy from a coarser
partition, so this script tests the claim at the level of item pairs:

  * every pair of Bayley-4 items is classified as
      consensus-together : both references put the two items together
      consensus-apart    : both references put them apart
      disputed           : the references disagree
  * for the detected clusters we report the share of consensus-together
    pairs kept together, of consensus-apart pairs kept apart, and, for the
    disputed pairs, how many are resolved the expert way vs the empirical way.

Three treatments of the three multi-factor items (COG_035, COG_048, COG_051)
are reported; COG_047 (no expert domain) and the five items outside the
empirical model (COG_002, 003, 008, 052, 081) are always excluded because a
pair needs a label in both references:

  strict      31 items: multi-factor items excluded
  any_factor  34 items: two items are "together" under the expert model if
                        they share any listed domain
  primary     34 items: multi-factor items carry their first-listed domain
                        (the resolution used for the transfer to the Bayley-III)

Writes `Bayley4_PairLevel_Intermediacy` to results/metrics/manuscript_metrics.json
and results/tables/b4_pair_level_intermediacy.csv.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):          # utils.update_results_json prints a check mark
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bayley_nlp"))
from utils import update_results_json  # noqa: E402

REF_YAML = ROOT / "config/reference_models_b4.yaml"
CMP_CSV = (ROOT / "data/outputs/step2_b4_validation"
           / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
           / "05_model_comparison/item_comparison_theoretical.csv")
OUT_CSV = ROOT / "results/tables/b4_pair_level_intermediacy.csv"
METRICS = ROOT / "results/metrics/manuscript_metrics.json"

ABBREV = {"attention": "ATT", "working_memory": "WM", "goal_directed_problem_solving": "GDPS",
          "flexibility_shift": "FS", "higher_order_processing": "HOP"}
DETECTED = {"attention": "ATT", "working memory": "WM", "goal directed problem solving": "GDPS",
            "flexibility shift": "FS", "higher order processes": "HOP",
            "higher order processing": "HOP"}


def load() -> tuple[dict, dict, dict, dict]:
    ref = yaml.safe_load(REF_YAML.read_text(encoding="utf-8"))
    theo_all: dict[str, list[str]] = {}
    for f in ABBREV:                                   # yaml order = first-wins order
        for i in ref["theoretical"].get(f, []):
            theo_all.setdefault(i, []).append(ABBREV[f])
    theo_prim = {i: v[0] for i, v in theo_all.items()}
    emp = {i: f for f, items in ref["empirical"].items()
           if f not in ("description", "source") for i in items}
    cmp = pd.read_csv(CMP_CSV)
    det = {r.item_id: DETECTED[str(r.detected_factor).strip().lower()] for r in cmp.itertuples()}
    return theo_all, theo_prim, emp, det


def pair_table(items, together_theo, emp, det) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(sorted(items), 2):
        rows.append({"item_a": a, "item_b": b,
                     "theo_together": bool(together_theo(a, b)),
                     "emp_together": emp[a] == emp[b],
                     "det_together": det[a] == det[b]})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    ct = df[df.theo_together & df.emp_together]
    ca = df[~df.theo_together & ~df.emp_together]
    dis = df[df.theo_together != df.emp_together]
    d_theo = dis[dis.theo_together]          # expert together, empirical apart
    d_emp = dis[dis.emp_together]            # empirical together, expert apart
    sided_expert = int(d_theo.det_together.sum()) + int((~d_emp.det_together).sum())
    sided_emp = int((~d_theo.det_together).sum()) + int(d_emp.det_together.sum())

    def jacc(x, y):
        return round(float((x & y).sum() / (x | y).sum()), 4)

    return {
        "n_pairs": int(len(df)),
        "consensus_together": {"n": int(len(ct)), "kept_together": int(ct.det_together.sum()),
                               "pct": round(100 * ct.det_together.mean(), 1)},
        "consensus_apart": {"n": int(len(ca)), "kept_apart": int((~ca.det_together).sum()),
                            "pct": round(100 * (~ca.det_together).mean(), 1)},
        "disputed": {"n": int(len(dis)),
                     "expert_together_pairs": int(len(d_theo)),
                     "expert_together_kept_together": int(d_theo.det_together.sum()),
                     "empirical_together_pairs": int(len(d_emp)),
                     "empirical_together_kept_together": int(d_emp.det_together.sum()),
                     "resolved_toward_expert": sided_expert,
                     "resolved_toward_empirical": sided_emp,
                     "pct_toward_expert": round(100 * sided_expert / len(dis), 1)},
        "together_pairs": {"expert": int(df.theo_together.sum()), "empirical": int(df.emp_together.sum()),
                           "detected": int(df.det_together.sum())},
        "jaccard": {"detected_vs_expert": jacc(df.det_together, df.theo_together),
                    "detected_vs_empirical": jacc(df.det_together, df.emp_together),
                    "expert_vs_empirical": jacc(df.theo_together, df.emp_together)},
    }


def main() -> None:
    theo_all, theo_prim, emp, det = load()
    multi = sorted(i for i, v in theo_all.items() if len(v) > 1)
    common = sorted(i for i in det if i in emp and i in theo_all)
    strict = [i for i in common if i not in multi]
    treatments = {
        "strict": (strict, lambda a, b: theo_prim[a] == theo_prim[b]),
        "any_factor": (common, lambda a, b: bool(set(theo_all[a]) & set(theo_all[b]))),
        "primary": (common, lambda a, b: theo_prim[a] == theo_prim[b]),
    }
    out: dict = {
        "description": "pair-level agreement of the detected Bayley-4 clusters with the expert-derived "
                       "and empirical Aylward models (see analysis/concordance/b4_pair_level_intermediacy.py)",
        "excluded_always": {"no_expert_domain": sorted(set(det) - set(theo_all)),
                            "not_in_empirical_model": sorted(set(det) - set(emp))},
        "multi_factor_items": {i: theo_all[i] for i in multi},
        "cluster_sizes": {"detected": pd.Series(det).value_counts().to_dict(),
                          "expert_primary": pd.Series(theo_prim).value_counts().to_dict(),
                          "empirical": pd.Series(emp).value_counts().to_dict()},
    }
    tables = []
    for name, (items, fn) in treatments.items():
        df = pair_table(items, fn, emp, det)
        out[name] = {"n_items": len(items), **summarise(df)}
        df.insert(0, "treatment", name)
        tables.append(df)
        s = out[name]
        print(f"[{name:<10s}] {len(items)} items, {s['n_pairs']} pairs: consensus-together kept "
              f"{s['consensus_together']['kept_together']}/{s['consensus_together']['n']} "
              f"({s['consensus_together']['pct']}%), consensus-apart kept apart "
              f"{s['consensus_apart']['kept_apart']}/{s['consensus_apart']['n']} "
              f"({s['consensus_apart']['pct']}%), disputed {s['disputed']['n']} -> expert "
              f"{s['disputed']['resolved_toward_expert']} / empirical "
              f"{s['disputed']['resolved_toward_empirical']}; together-pairs expert/empirical/detected "
              f"{s['together_pairs']['expert']}/{s['together_pairs']['empirical']}/{s['together_pairs']['detected']}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(OUT_CSV, index=False)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    update_results_json("Bayley4_PairLevel_Intermediacy", out, filename=str(METRICS))
    print(f"Pair table -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
