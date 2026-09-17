#!/usr/bin/env python
"""
Build the final 91-item Bayley-III domain taxonomy (Objective 1 deliverable).

The manuscript's domain structure for the Bayley-III cognition scale is
assembled from three committed pipeline outputs, in this order of precedence:

1. ``data/outputs/step1_translation/b4_b3_cascade_pairs.csv``
   39 items whose domain is inherited from their Bayley-4 counterpart
   (Aylward et al., 2022) through cascade semantic pairing.
2. ``data/outputs/step1_translation/mixed_centroids/all_unpaired_b3_assignments.csv``
   52 items assigned to the nearest domain centroid.  Items whose closest
   centroid was the unclassified cluster (UC) are redirected to the best
   domain (``best_ef_factor``); those with ``uc_margin`` > 0.10 were flagged
   for expert review.
3. ``expert_review/records/coauthor_review_b3_solution.xlsx``
   (sheet "UC Uncertain (Step 1)") -- the two experts' independent
   assignments for the flagged items.  Where the experts agreed with each
   other and their consensus differs from the algorithm, the consensus
   overrides the algorithmic label (Methods, Objective 1).

The script writes ``results/tables/bayley3_item_domain_assignments.csv``,
cross-checks it against the model specification committed by
``expert_review/import_review_b3.py`` (``config/model_specs.yaml`` key
``all_mpnet_base_v2_b3_coauthor_consensus``) and against the numbers reported
in the paper (ATT 22, WM 19, GDPS 23, FS 20, HOP 7; 18/23 GDPS, 13/22 ATT,
7/19 WM and 3/7 HOP items without a Bayley-4 counterpart), and prints the
per-domain counts.

Every downstream figure and metric that needs the "translated structure"
(Figure 2, Figure 5B, Figures S4/S7, the concordance metrics) reads this table
so that the paper, the supplement and the interactive explorer cannot drift
apart.

Usage (from the repository root):
    python analysis/item_assignments.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

PAIRS_CSV = ROOT / "data/outputs/step1_translation/b4_b3_cascade_pairs.csv"
UNPAIRED_CSV = ROOT / "data/outputs/step1_translation/mixed_centroids/all_unpaired_b3_assignments.csv"
REVIEW_XLSX = ROOT / "expert_review/records/coauthor_review_b3_solution.xlsx"
REVIEW_SHEET = "UC Uncertain (Step 1)"
MODEL_SPECS = ROOT / "config/model_specs.yaml"
FINAL_SPEC_KEY = "all_mpnet_base_v2_b3_coauthor_consensus"
CFA_INP = ROOT / "psychometrics/measurement/model_cfa_3factor.inp"
FACTOR_ITEM_MAP = ROOT / "psychometrics/factor_item_map.csv"
OUT_CSV = ROOT / "results/tables/bayley3_item_domain_assignments.csv"
OUT_COUNTS = ROOT / "results/tables/bayley3_domain_counts.csv"

UC_MARGIN_THRESHOLD = 0.10          # Methods: flagged for expert review if sim(UC) - sim(best domain) > .10
UC_LABEL = "uncategorized_cognition"

SHORT = {
    "attention": "ATT",
    "working_memory": "WM",
    "goal_directed_problem_solving": "GDPS",
    "flexibility_shift": "FS",
    "higher_order_processing": "HOP",
}
LONG = {v: k for k, v in SHORT.items()}
ORDER = ["ATT", "WM", "GDPS", "FS", "HOP"]

# Numbers reported in the manuscript (Results, "Translating the theoretical
# domain structure"; Discussion, "Embedding-based domain differentiation").
PAPER_COUNTS = {"ATT": 22, "WM": 19, "GDPS": 23, "FS": 20, "HOP": 7}
PAPER_EXTENDED = {"GDPS": 18, "ATT": 13, "WM": 7, "HOP": 3}   # items without a Bayley-4 counterpart
PAPER_N_FLAGGED = 6


def _short(label) -> str | None:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    s = str(label).strip()
    if s in SHORT:
        return SHORT[s]
    key = re.sub(r"[\s()/\-]+", "", s).upper()
    return key if key in ORDER else None


def load_expert_review() -> pd.DataFrame:
    """Return the expert assignments for the UC-flagged items (one row per item)."""
    df = pd.read_excel(REVIEW_XLSX, sheet_name=REVIEW_SHEET)
    df = df[df["Item ID"].astype(str).str.startswith("COG_")].copy()
    cols = {c: c.split("\n")[0].strip() for c in df.columns}
    df = df.rename(columns=cols)
    df["expert_1"] = df["Expert AD"].map(_short)
    df["expert_2"] = df["Expert AH"].map(_short)
    df["expert_consensus"] = [
        a if (a is not None and a == b) else None for a, b in zip(df["expert_1"], df["expert_2"])
    ]
    return df[["Item ID", "expert_1", "expert_2", "expert_consensus"]].rename(
        columns={"Item ID": "item_id"})


def build() -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS_CSV)
    unpaired = pd.read_csv(UNPAIRED_CSV)
    review = load_expert_review().set_index("item_id")

    rows = []
    # 1. cascade-paired items ------------------------------------------------
    for r in pairs.itertuples():
        rows.append({
            "item_id": r.b3_item_id,
            "item_title": r.b3_item_title,
            "domain": SHORT[r.b4_factor],
            "assignment_source": "cascade_pairing",
            "pairing_stage": {"title_locked": "title-locked", "fulltext": "full-text fallback"}[r.pairing_stage],
            "b4_item_id": r.b4_item_id,
            "b4_item_title": r.b4_item_title,
            "similarity_title": round(float(r.sim_title), 4),
            "similarity_fulltext": round(float(r.sim_fulltext), 4),
            "algorithmic_domain": SHORT[r.b4_factor],
            "uc_flagged": False,
        })
    # 2. centroid-assigned items --------------------------------------------
    for r in unpaired.itertuples():
        redirected = r.assigned_factor == UC_LABEL
        algo = SHORT[r.best_ef_factor if redirected else r.assigned_factor]
        flagged = bool(redirected and r.uc_margin > UC_MARGIN_THRESHOLD)
        source = "centroid_uc_redirected" if redirected else "centroid_nearest_domain"
        final = algo
        e1 = e2 = cons = None
        if flagged:
            if r.item_id not in review.index:
                raise RuntimeError(f"{r.item_id} was flagged for expert review but is not in {REVIEW_XLSX.name}")
            e1, e2, cons = (None if pd.isna(v) else v for v in
                            review.loc[r.item_id, ["expert_1", "expert_2", "expert_consensus"]])
            if cons is not None and cons != algo:
                final, source = cons, "expert_review_override"
            elif cons is not None:
                source = "expert_review_confirmed"
            else:
                source = "expert_review_no_consensus"
        rows.append({
            "item_id": r.item_id,
            "item_title": r.item_title,
            "domain": final,
            "assignment_source": source,
            "pairing_stage": None,
            "b4_item_id": None,
            "b4_item_title": None,
            "similarity_title": None,
            "similarity_fulltext": None,
            "algorithmic_domain": algo,
            "uc_flagged": flagged,
            "uc_margin": round(float(r.uc_margin), 4),
            "sim_ATT": round(float(r.sim_ATT), 4), "sim_WM": round(float(r.sim_WM), 4),
            "sim_GDPS": round(float(r.sim_GDPS), 4), "sim_FS": round(float(r.sim_FS), 4),
            "sim_HOP": round(float(r.sim_HOP), 4), "sim_UC": round(float(r.sim_UC), 4),
            "expert_1_AD": e1, "expert_2_AH": e2, "expert_consensus": cons,
        })

    df = pd.DataFrame(rows)
    df["item_number"] = df["item_id"].str.extract(r"(\d+)$").astype(int)
    df["domain_name"] = df["domain"].map(LONG)

    # 3. empirical-model membership ------------------------------------------
    df["in_dhcp_range"] = df["item_number"].between(34, 68)
    fmap = pd.read_csv(FACTOR_ITEM_MAP)
    in_mplus_data = set(fmap["item_num"].astype(int))
    inp = CFA_INP.read_text(encoding="utf-8", errors="ignore")
    use = re.search(r"USEVARIABLES ARE(.*?);", inp, flags=re.S).group(1)
    cfa_items = {int(m) for m in re.findall(r"COG(\d{3})", use)}
    df["in_mplus_dataset"] = df["item_number"].isin(in_mplus_data)
    df["in_cfa_model"] = df["item_number"].isin(cfa_items)

    cols = ["item_id", "item_number", "item_title", "domain", "domain_name", "assignment_source",
            "algorithmic_domain", "uc_flagged", "uc_margin", "expert_1_AD", "expert_2_AH",
            "expert_consensus", "pairing_stage", "b4_item_id", "b4_item_title",
            "similarity_title", "similarity_fulltext",
            "sim_ATT", "sim_WM", "sim_GDPS", "sim_FS", "sim_HOP", "sim_UC",
            "in_dhcp_range", "in_mplus_dataset", "in_cfa_model"]
    df = df[cols].sort_values("item_number").reset_index(drop=True)
    return df


def check(df: pd.DataFrame) -> None:
    assert len(df) == 91 and df["item_id"].is_unique, "expected 91 unique Bayley-III items"
    counts = Counter(df["domain"])
    ext = Counter(df.loc[df["assignment_source"] != "cascade_pairing", "domain"])
    n_flagged = int(df["uc_flagged"].sum())
    n_override = int((df["assignment_source"] == "expert_review_override").sum())

    # against config/model_specs.yaml (written by expert_review/import_review_b3.py)
    spec = yaml.safe_load(MODEL_SPECS.read_text(encoding="utf-8"))[FINAL_SPEC_KEY]
    spec_map = {iid: SHORT[spec["factor_names"][f]] for f, items in spec["factor_items"].items()
                for iid in items}
    mismatch = {i: (d, spec_map.get(i)) for i, d in zip(df["item_id"], df["domain"]) if spec_map.get(i) != d}
    assert not mismatch, f"disagreement with {FINAL_SPEC_KEY}: {mismatch}"

    problems = []
    if {k: counts[k] for k in ORDER} != PAPER_COUNTS:
        problems.append(f"domain counts {dict(counts)} != paper {PAPER_COUNTS}")
    if {k: ext[k] for k in PAPER_EXTENDED} != PAPER_EXTENDED:
        problems.append(f"extended-item counts {dict(ext)} != paper {PAPER_EXTENDED}")
    if n_flagged != PAPER_N_FLAGGED:
        problems.append(f"{n_flagged} UC-flagged items != paper {PAPER_N_FLAGGED}")
    if problems:
        raise AssertionError("; ".join(problems))

    print("Final 91-item Bayley-III domain taxonomy")
    print("  per domain      :", {k: counts[k] for k in ORDER}, "(paper: ATT 22, WM 19, GDPS 23, FS 20, HOP 7)")
    print("  without B4 pair :", {k: ext[k] for k in ORDER}, "(paper: 18/23 GDPS, 13/22 ATT, 7/19 WM, 3/7 HOP)")
    print(f"  UC-flagged for expert review: {n_flagged}; expert consensus overrode the algorithm for "
          f"{n_override} item(s): "
          + ", ".join(f"{r.item_id} {r.algorithmic_domain}->{r.domain}"
                      for r in df[df["assignment_source"] == "expert_review_override"].itertuples()))
    print(f"  items 34-68 (dHCP range): {int(df['in_dhcp_range'].sum())}; in Mplus data set: "
          f"{int(df['in_mplus_dataset'].sum())}; retained in the 23-item CFA: {int(df['in_cfa_model'].sum())}")
    print("  matches config/model_specs.yaml key", FINAL_SPEC_KEY)


def main() -> int:
    df = build()
    check(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    counts = (df.groupby("domain")["assignment_source"].value_counts().unstack(fill_value=0)
              .reindex(ORDER))
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(OUT_COUNTS)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} and {OUT_COUNTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
