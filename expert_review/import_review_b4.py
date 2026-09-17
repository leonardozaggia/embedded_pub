#!/usr/bin/env python
"""
Import the Bayley-4 (Step-2) co-author review and compute Cohen's kappa.

This is a VALIDATION-ONLY companion to import_review_b3.py: it
quantifies how well the two experts agree with each other and with the
reference / detected structures on the divergent Bayley-4 items. It never
writes a model spec — Step 2 only validates the NLP approach, it does not
build the measurement model.

Agreement metrics computed
--------------------------
    Expert 1 (A.D.) vs Expert 2 (A.H.)          inter-rater reliability
    Consensus       vs Aylward reference        do experts back the reference?
    Consensus       vs Detected (bottom-up NLP) do experts back the machine?
    Expert 1        vs Aylward reference
    Expert 2        vs Aylward reference

Consensus is derived where the two experts agree (primary factor); unresolved
disagreements are left out of the consensus comparisons.

Usage
-----
    python expert_review/import_review_b4.py
    python expert_review/import_review_b4.py --input path/to/coauthor_review_b4.xlsx
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")
del _s

import argparse
import json
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(Path(__file__).parent))
# Reuse the B3 importer's normalisation + kappa/confusion helpers.
import import_review_b3 as imp3   # noqa: E402

DEFAULT_INPUT = _REPO / "expert_review/records/coauthor_review_b4.xlsx"
KAPPA_JSON    = _REPO / "expert_review/records/coauthor_review_b4_kappa.json"
KAPPA_CSV     = _REPO / "expert_review/records/coauthor_review_b4_kappa.csv"
SHEET_HINT    = "discordant"


def load_b4_review(path: Path) -> pd.DataFrame:
    """Return item_id, expert1, expert2, consensus, reference, detected."""
    wb = pd.read_excel(path, sheet_name=None)
    sheet = next((s for s in wb if SHEET_HINT in s.lower()), None)
    if sheet is None:
        raise ValueError(f"No '{SHEET_HINT}...' sheet found in {path.name}")
    df = wb[sheet].rename(columns={c: str(c).replace("\n", " ").strip()
                                   for c in wb[sheet].columns})

    def col(*cands):
        for cand in cands:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
        return None

    id_c   = col("item id", "item_id")
    e1_c   = col("expert 1")
    e2_c   = col("expert 2")
    cons_c = col("consensus")
    ref_c  = col("aylward", "reference", "theoretical")
    det_c  = col("detected", "bottom-up", "nlp")

    rows = []
    for _, r in df.iterrows():
        if id_c is None or pd.isna(r.get(id_c)):
            continue
        e1 = imp3.primary_factor(r.get(e1_c)) if e1_c else None
        e2 = imp3.primary_factor(r.get(e2_c)) if e2_c else None
        # Prefer a hand-entered consensus; otherwise derive from agreement.
        cons = imp3.primary_factor(r.get(cons_c)) if cons_c else None
        if cons is None and e1 is not None and e1 == e2:
            cons = e1
        rows.append({
            "item_id":   str(r[id_c]).strip(),
            "expert1":   e1,
            "expert2":   e2,
            "consensus": cons,
            "reference": imp3.primary_factor(r.get(ref_c)) if ref_c else None,
            "detected":  imp3.primary_factor(r.get(det_c)) if det_c else None,
        })
    return pd.DataFrame(rows)


def main(input_path: Path, dry_run: bool) -> None:
    print("=" * 65)
    print("  IMPORT CO-AUTHOR REVIEW — B4 (Step 2, validation only)")
    print("=" * 65)
    print(f"  Input  : {input_path}\n")

    review = load_b4_review(input_path)
    n = len(review)
    print(f"  {n} divergent items loaded.")
    print(f"  Expert 1 filled : {review['expert1'].notna().sum()} / {n}")
    print(f"  Expert 2 filled : {review['expert2'].notna().sum()} / {n}")
    print(f"  Consensus (derived where experts agree): "
          f"{review['consensus'].notna().sum()} / {n}\n")

    comparisons = [
        ("Expert 1 vs Expert 2 (IRR)",          "expert1",   "expert2"),
        ("Consensus vs Aylward (reference)",    "consensus", "reference"),
        ("Consensus vs Detected (NLP)",         "consensus", "detected"),
        ("Expert 1 vs Aylward (reference)",     "expert1",   "reference"),
        ("Expert 2 vs Aylward (reference)",     "expert2",   "reference"),
    ]
    results: dict[str, float | None] = {}
    for label, a, b in comparisons:
        k = imp3._kappa(review[a].tolist(), review[b].tolist(), label)
        results[label] = k
        if k is not None:
            print(f"   {label:<38s} κ = {k:.4f}")

    # Confusion: consensus vs reference and vs detected.
    for ref_col, title in [("reference", "Consensus (rows) vs Aylward (cols)"),
                           ("detected",  "Consensus (rows) vs Detected NLP (cols)")]:
        cm = imp3._confusion(review["consensus"].tolist(), review[ref_col].tolist())
        if not cm.empty:
            imp3.print_confusion(cm, title)

    # Classify the workbook rows.  The workbook was generated from the
    # `agrees` column of item_comparison_theoretical.csv, which at the time
    # flagged COG_048 (GDPS/HOP, clustered as HOP) as discordant because of a
    # label-string mismatch; neither expert rated it.  Under the any-factor
    # rule the discordant set is the 16 items whose cluster matches none of
    # their expert domains (15) or that have no expert domain (COG_047).
    rated = review[review["expert1"].notna() | review["expert2"].notna()]
    ref_sets = review["reference"].fillna("unassigned").astype(str).str.split(",")
    discordant_mask = [
        (ref == ["unassigned"]) or (det not in [x.strip() for x in ref])
        for ref, det in zip(ref_sets, review["detected"].astype(str))
    ]
    discordant = review.loc[discordant_mask, "item_id"].tolist()
    payload = {
        "input": input_path.name,
        "n_items": int(n),
        "n_expert1": int(review["expert1"].notna().sum()),
        "n_expert2": int(review["expert2"].notna().sum()),
        "n_consensus": int(review["consensus"].notna().sum()),
        "n_rated_items": int(len(rated)),
        "n_discordant_any_factor": int(len(discordant)),
        "items_reviewed": sorted(review["item_id"].tolist()),
        "items_unrated": sorted(set(review["item_id"]) - set(rated["item_id"])),
        "items_reviewed_but_concordant": sorted(set(review["item_id"]) - set(discordant)),
        "items_without_reference_domain": sorted(
            review.loc[review["reference"].fillna("unassigned") == "unassigned", "item_id"].tolist()),
        "items_with_consensus": sorted(review.loc[review["consensus"].notna(), "item_id"].tolist()),
        "kappa": {k: (None if v is None else round(float(v), 4))
                  for k, v in results.items()},
    }
    if dry_run:
        print("\n  [dry-run] Metrics (not written):\n  "
              + json.dumps(payload, ensure_ascii=False))
    else:
        with open(KAPPA_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        pd.DataFrame([{"comparison": k, "kappa": v}
                      for k, v in results.items()]).to_csv(KAPPA_CSV, index=False)
        print(f"\n  Agreement metrics written to:\n    {KAPPA_JSON}\n    {KAPPA_CSV}")

    print("\n" + "=" * 65)
    print("  DONE — no model spec written (Step 2 is validation-only).")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import B4 (Step-2) co-author review; compute Cohen's kappa."
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                        help=f"B4 review workbook (default: {DEFAULT_INPUT.name}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print metrics instead of writing them.")
    args = parser.parse_args()
    main(input_path=args.input, dry_run=args.dry_run)
