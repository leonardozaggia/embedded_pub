#!/usr/bin/env python
"""
Import co-author review assignments from the returned Excel workbook.

Workflow
--------
1. Read returned xlsx (Expert 1, Expert 2, Consensus columns across both sheets).
2. Match items back to item_ids (by Item ID column if present, else by item title).
3. Compute Cohen's kappa:
     - Expert 1 vs Expert 2  (inter-rater reliability)
     - Expert 1 vs Step 3 detected
     - Expert 2 vs Step 3 detected
     - Consensus vs Step 3 detected
     - Consensus vs Step 1 theoretical
4. Distinguish structural label-swaps (whole-cluster relabeling) from genuine
   single-item relocations by comparing confusion matrix off-diagonals.
5. Build a full 91-item model:
     - Reviewed items  → use Consensus assignment
     - Unreviewed items → inherit Step 1 theoretical assignment
6. Save as a new entry in config/model_specs.yaml.

Usage
-----
    python expert_review/import_review_b3.py
    python expert_review/import_review_b3.py --input path/to/returned.xlsx
    python expert_review/import_review_b3.py --model-name my_b3_coauthor_v1
    python expert_review/import_review_b3.py --dry-run
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")
del _s

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT   = _REPO / "expert_review/records/coauthor_review_b3_solution.xlsx"
DEFAULT_MODEL   = "all_mpnet_base_v2_b3_coauthor_consensus"
MODEL_SPECS_YAML = _REPO / "config/model_specs.yaml"

STEP1_DIR  = _REPO / "data/outputs/step1_translation"
STEP3_DIR  = _REPO / "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"
ITEMS_CSV  = STEP3_DIR / "items.csv"

CENTROID_METHOD = "mixed_centroids"

# ---------------------------------------------------------------------------
# Factor label normalisation
# ---------------------------------------------------------------------------
_ABBREV_MAP: dict[str, str] = {
    "att":   "attention",
    "wm":    "working_memory",
    "gdps":  "goal_directed_problem_solving",
    "fs":    "flexibility_shift",
    "hop":   "higher_order_processing",
}

_DISPLAY_MAP: dict[str, str] = {
    "attention (att)":                     "attention",
    "working memory (wm)":                 "working_memory",
    "goal-directed problem solving (gdps)":"goal_directed_problem_solving",
    "flexibility / shift (fs)":            "flexibility_shift",
    "flexibility/shift (fs)":              "flexibility_shift",
    "higher-order processing (hop)":       "higher_order_processing",
    "higher order processing (hop)":       "higher_order_processing",
}

_CANONICAL: dict[str, str] = {
    "higher_order_processes":          "higher_order_processing",
    "goal directed problem solving":   "goal_directed_problem_solving",
    "flexibility shift":               "flexibility_shift",
    "working memory":                  "working_memory",
}


def normalize_factor(raw) -> str | None:
    """Return canonical snake_case factor name, or None if blank/unparseable."""
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    s = str(raw).strip().lower()

    if s in _DISPLAY_MAP:
        return _DISPLAY_MAP[s]

    # Try abbreviation alone (ATT, WM, …)
    key = re.sub(r"[\s\(\)/\-]+", "", s)
    if key in _ABBREV_MAP:
        return _ABBREV_MAP[key]

    # Generic snake_case normalisation
    normed = re.sub(r"[\s\-]+", "_", s)
    return _CANONICAL.get(normed, normed)


# Factor-abbreviation tokens, longest-first so "GDPS" is matched before "G".
_FACTOR_TOKENS = ["gdps", "att", "wm", "fs", "hop"]


def primary_factor(raw) -> str | None:
    """
    Reduce an expert answer to its PRIMARY (first-listed) cognitive factor.

    Experts sometimes give multi-factor answers ("ATT, WM", "HOP with GDPS as
    first stage", "GDPS and HOP"). Cohen's kappa and the confusion matrices
    require a single label per item, so we take the first factor token that
    appears in the string, left to right. Pure single-factor answers pass
    through unchanged. Returns canonical snake_case, or None if blank/unparseable.
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    s = str(raw).strip().lower()
    # Scan word tokens left→right for the first recognised factor abbreviation.
    for tok in re.findall(r"[a-z]+", s):
        if tok in _ABBREV_MAP:
            return _ABBREV_MAP[tok]
    # Fall back to the general normaliser (handles display/canonical names).
    return normalize_factor(raw)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_items() -> pd.DataFrame:
    df = pd.read_csv(ITEMS_CSV)
    df["item_title_lower"] = df["item_title"].str.strip().str.lower()
    return df[["item_id", "item_title", "item_title_lower"]]


def load_step1_assignments() -> pd.DataFrame:
    """All 91 B3 items with step1_factor (UC resolved to best_ef_factor)."""
    paired_csv   = STEP1_DIR / "b3_paired_assignments.csv"
    unpaired_csv = STEP1_DIR / CENTROID_METHOD / "all_unpaired_b3_assignments.csv"

    paired = pd.read_csv(paired_csv).rename(
        columns={"assigned_factor": "step1_factor"}
    )
    paired["step1_factor"] = paired["step1_factor"].apply(normalize_factor)

    unpaired = pd.read_csv(unpaired_csv)
    unpaired["step1_factor"] = unpaired.apply(
        lambda r: normalize_factor(
            r["best_ef_factor"]
            if r["assigned_factor"] == "uncategorized_cognition"
            else r["assigned_factor"]
        ),
        axis=1,
    )
    return pd.concat(
        [paired[["item_id", "step1_factor"]], unpaired[["item_id", "step1_factor"]]],
        ignore_index=True,
    )


def load_step3_assignments() -> pd.DataFrame:
    """91 B3 items with step3_factor from bottom-up NLP clustering."""
    clust  = pd.read_csv(STEP3_DIR / "03_clustering/cluster_assignments.csv")
    labels = pd.read_csv(STEP3_DIR / "04_labeling/cluster_concept_labels.csv")
    id_to_name = dict(zip(labels["cluster_id"], labels["top_concept"]))
    clust["step3_factor"] = clust["cluster"].map(id_to_name).apply(normalize_factor)
    return clust[["item_id", "step3_factor"]]


def load_review_xlsx(path: Path, items: pd.DataFrame) -> pd.DataFrame:
    """
    Read the returned review workbook and return a DataFrame with columns:
        item_id, expert1, expert2, consensus, sheet
    Items are matched by 'Item ID' column if present, otherwise by title.
    """
    wb = pd.read_excel(path, sheet_name=None)

    title_to_id = dict(zip(items["item_title_lower"], items["item_id"]))
    records = []

    for sheet_name, df in wb.items():
        if sheet_name.lower() == "legend":
            continue

        # Normalise column names for lookup
        col_map = {c: c.replace("\n", " ").strip() for c in df.columns}
        df = df.rename(columns=col_map)

        def _find_col(*candidates) -> str | None:
            for cand in candidates:
                for col in df.columns:
                    if cand.lower() in col.lower():
                        return col
            return None

        id_col       = _find_col("item id", "item_id")
        title_col    = _find_col("item title", "item_title")
        # "Expert AD"/"Expert AH" (this workbook's actual headers) must be
        # checked before the generic "expert 1"/"expert 2" fallback, else a
        # header like "Expert AD Assignment" is never matched.
        expert1_col  = _find_col("expert ad", "expert 1")
        expert2_col  = _find_col("expert ah", "expert 2")
        consensus_col = _find_col("consensus")

        if title_col is None:
            print(f"  [warn] Sheet '{sheet_name}': no item title column — skipped.")
            continue

        for _, row in df.iterrows():
            # Skip fully-blank trailing rows (no id AND no title).
            if (id_col is None or pd.isna(row.get(id_col))) and pd.isna(row.get(title_col)):
                continue

            # Resolve item_id
            if id_col and pd.notna(row.get(id_col)):
                item_id = str(row[id_col]).strip()
            else:
                title_key = str(row[title_col]).strip().lower()
                item_id = title_to_id.get(title_key)
                if item_id is None:
                    print(f"  [warn] Could not match title '{row[title_col]}' — skipped.")
                    continue

            e1 = primary_factor(row.get(expert1_col)) if expert1_col else None
            e2 = primary_factor(row.get(expert2_col)) if expert2_col else None
            cons = primary_factor(row.get(consensus_col)) if consensus_col else None
            # The Consensus column is only filled in for manually-adjudicated
            # disagreements; where both experts already agree, derive it.
            if cons is None and e1 is not None and e1 == e2:
                cons = e1

            records.append({
                "item_id":   item_id,
                "expert1":   e1,
                "expert2":   e2,
                "consensus": cons,
                "sheet":     sheet_name,
                "is_uc":     _is_uc_sheet(sheet_name),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # An item can appear in BOTH the UC and discordant sheets; UC membership
    # (which grants model-override eligibility) must survive de-duplication.
    df["is_uc"] = df.groupby("item_id")["is_uc"].transform("max").astype(bool)
    return df.drop_duplicates(subset=["item_id"])


def _load_one_expert(path: Path, items: pd.DataFrame) -> dict[str, dict]:
    """
    Read a single-expert workbook (per-reviewer format: 'Item Title' +
    'Expert Assignment' columns, one sheet per item batch) and return
        {item_id: {"factor": <canonical|None>, "sheet": <sheet_name>}}.
    Matches items by title.
    """
    title_to_id = dict(zip(items["item_title_lower"], items["item_id"]))
    out: dict[str, dict] = {}
    wb = pd.read_excel(path, sheet_name=None)
    for sheet_name, df in wb.items():
        if sheet_name.strip().lower() in ("legend", "_placeholder"):
            continue
        tcol = next((c for c in df.columns if "title"  in str(c).lower()), None)
        acol = next((c for c in df.columns if "assign" in str(c).lower()), None)
        if tcol is None or acol is None:
            continue
        for _, row in df.iterrows():
            item_id = title_to_id.get(str(row[tcol]).strip().lower())
            if item_id is None:
                continue
            factor = primary_factor(row.get(acol))
            is_uc = _is_uc_sheet(sheet_name)
            prev = out.get(item_id)
            if prev is None:
                out[item_id] = {"factor": factor, "sheet": sheet_name, "is_uc": is_uc}
            else:
                # First non-empty factor wins; UC membership is OR'd across sheets.
                if prev["factor"] is None and factor is not None:
                    prev["factor"], prev["sheet"] = factor, sheet_name
                prev["is_uc"] = prev["is_uc"] or is_uc
    return out


def load_review_per_expert(expert1_path: Path, expert2_path: Path,
                           items: pd.DataFrame) -> pd.DataFrame:
    """
    Build the standard review DataFrame (item_id, expert1, expert2, consensus,
    sheet) from two per-expert workbooks. Consensus is auto-derived only where
    both experts agree; disagreements are left None (unresolved) until the
    experts reconcile — matching the Methods' 'discussed until consensus'.
    """
    e1 = _load_one_expert(expert1_path, items)
    e2 = _load_one_expert(expert2_path, items) if expert2_path.exists() else {}
    records = []
    for item_id in sorted(set(e1) | set(e2)):
        f1 = e1.get(item_id, {}).get("factor")
        f2 = e2.get(item_id, {}).get("factor")
        consensus = f1 if (f1 is not None and f1 == f2) else None
        d1, d2 = e1.get(item_id, {}), e2.get(item_id, {})
        records.append({
            "item_id":   item_id,
            "expert1":   f1,
            "expert2":   f2,
            "consensus": consensus,
            "sheet":     d1.get("sheet", d2.get("sheet")),
            "is_uc":     bool(d1.get("is_uc") or d2.get("is_uc")),
        })
    return pd.DataFrame(records).drop_duplicates(subset=["item_id"])


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

FACTOR_ORDER = [
    "attention",
    "working_memory",
    "goal_directed_problem_solving",
    "flexibility_shift",
    "higher_order_processing",
]


def _is_uc_sheet(sheet_name) -> bool:
    """
    True for the Step-1 UC-ambiguous sheet only.

    These are the *only* reviewed items whose expert consensus is allowed to
    override the frozen theoretical model (Methods: expert assignment
    "overrode the algorithmic one"). The discordant sheet must NOT match —
    note its name literally contains "Step 1" ("Discordant (Step 3 vs Step 1)"),
    so we match on 'uncertain' / a leading 'uc' / '6 item' instead.
    """
    s = str(sheet_name).strip().lower()
    return ("uncertain" in s) or s.startswith("uc") or ("6 item" in s)


def _kappa(y_true: list, y_pred: list, label: str) -> float | None:
    """Compute Cohen's kappa; return None if insufficient data."""
    # Missing labels may be None or NaN depending on the pandas version, so
    # test with pd.isna rather than `is not None` (a NaN would otherwise be
    # counted as a category of its own and distort kappa).
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if not pd.isna(a) and not pd.isna(b)]
    if len(pairs) < 2:
        print(f"  [warn] {label}: only {len(pairs)} complete pairs — kappa skipped.")
        return None
    a, b = zip(*pairs)
    try:
        return cohen_kappa_score(a, b)
    except ValueError as e:
        print(f"  [warn] {label}: {e}")
        return None


def _confusion(y_true: list, y_pred: list) -> pd.DataFrame:
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if not pd.isna(a) and not pd.isna(b)]
    if not pairs:
        return pd.DataFrame()
    a, b = zip(*pairs)
    labels = sorted(set(a) | set(b), key=lambda x: FACTOR_ORDER.index(x) if x in FACTOR_ORDER else 99)
    cm = confusion_matrix(a, b, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def _detect_structural_swaps(confusion_df: pd.DataFrame, threshold: float = 0.70) -> list[tuple[str, str]]:
    """
    Return pairs of factors (row, col) where > threshold of row items landed in col.
    These are likely whole-cluster label swaps rather than genuine item relocations.
    """
    swaps = []
    for row_label in confusion_df.index:
        row = confusion_df.loc[row_label]
        total = row.sum()
        if total == 0:
            continue
        for col_label in confusion_df.columns:
            if row_label == col_label:
                continue
            if row[col_label] / total >= threshold:
                swaps.append((row_label, col_label))
    return swaps


def print_confusion(cm: pd.DataFrame, title: str) -> None:
    short = {
        "attention":                     "ATT",
        "working_memory":                "WM",
        "goal_directed_problem_solving": "GDPS",
        "flexibility_shift":             "FS",
        "higher_order_processing":       "HOP",
    }
    cm_short = cm.rename(index=short, columns=short)
    print(f"\n  {title}")
    print("  " + cm_short.to_string().replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Model spec builder
# ---------------------------------------------------------------------------

def build_model_spec(
    full_91: pd.DataFrame,
    model_name: str,
    n_reviewed: int,
    n_consensus_used: int,
    n_step1_fallback: int,
) -> dict:
    """Build a model_specs.yaml entry from a full 91-item assignment DataFrame."""
    factor_items: dict[str, list[str]] = defaultdict(list)
    for _, row in full_91.sort_values("item_id").iterrows():
        f = row["final_factor"]
        if f is None:
            continue
        # Map canonical name → factor key F1..F5
        factor_items[f].append(row["item_id"])

    # Sort factors by FACTOR_ORDER
    ordered = sorted(
        factor_items.keys(),
        key=lambda x: FACTOR_ORDER.index(x) if x in FACTOR_ORDER else 99,
    )
    fkeys = {name: f"F{i+1}" for i, name in enumerate(ordered)}

    factor_items_keyed = {fkeys[name]: sorted(items) for name, items in factor_items.items()}
    factor_names_keyed = {fkeys[name]: name for name in ordered}

    return {
        "source": "coauthor_consensus_review",
        "model": "all-mpnet-base-v2",
        "date": str(date.today()),
        "reference_model": "Coauthor consensus review of B3 item assignments",
        "n_factors": len(ordered),
        "n_items": int(full_91["final_factor"].notna().sum()),
        "centroid_method": CENTROID_METHOD,
        "n_reviewed_items": n_reviewed,
        "n_consensus_used": n_consensus_used,
        "n_step1_fallback": n_step1_fallback,
        "factor_names": factor_names_keyed,
        "factor_items": factor_items_keyed,
    }


# ---------------------------------------------------------------------------
# YAML save
# ---------------------------------------------------------------------------

def save_model_spec(spec: dict, model_name: str, dry_run: bool) -> None:
    with open(MODEL_SPECS_YAML, encoding="utf-8") as fh:
        existing = yaml.safe_load(fh) or {}

    if model_name in existing:
        print(f"\n  [warn] '{model_name}' already exists in model_specs.yaml — overwriting.")

    existing[model_name] = spec

    if dry_run:
        print(f"\n  [dry-run] Would write spec '{model_name}' to {MODEL_SPECS_YAML}")
        print("  Spec preview:")
        print("  " + yaml.dump({model_name: spec}, allow_unicode=True,
                                default_flow_style=False, sort_keys=False
                                ).replace("\n", "\n  "))
        return

    with open(MODEL_SPECS_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(existing, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n  Saved '{model_name}' to {MODEL_SPECS_YAML}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path, model_name: str, dry_run: bool,
         from_per_expert: bool = False,
         expert1_file: Path | None = None,
         expert2_file: Path | None = None) -> None:
    print("=" * 65)
    print("  IMPORT CO-AUTHOR REVIEW — B3 ITEMS")
    print("=" * 65)
    print(f"  Input  : {input_path}")
    print(f"  Model  : {model_name}")
    print(f"  Dry-run: {dry_run}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading reference data …")
    items = load_items()
    step1 = load_step1_assignments()
    step3 = load_step3_assignments()

    expert1_file = expert1_file or (_REPO / "expert_review/records/coauthor_review_b3_AD.xlsx")
    expert2_file = expert2_file or (_REPO / "expert_review/records/coauthor_review_b3_AH.xlsx")

    print("[2/5] Reading review workbook …")
    review = load_review_xlsx(input_path, items)
    # Auto-fallback: the *_solution.xlsx template ships blank; the real answers
    # live in the per-expert workbooks. Use them if the solution file is empty
    # (or when explicitly requested).
    solution_empty = (review["expert1"].notna().sum() == 0
                      and review["consensus"].notna().sum() == 0)
    if from_per_expert or solution_empty:
        reason = "requested" if from_per_expert else "solution workbook is blank"
        print(f"       Reading per-expert workbooks ({reason}):")
        print(f"         Expert 1 (A.D.): {expert1_file.name}")
        print(f"         Expert 2 (A.H.): {expert2_file.name}"
              + ("" if expert2_file.exists() else "  [missing]"))
        review = load_review_per_expert(expert1_file, expert2_file, items)
    print(f"       {len(review)} reviewed items found across both sheets.")

    has_expert1 = review["expert1"].notna().sum()
    has_expert2 = review["expert2"].notna().sum()
    has_consensus = review["consensus"].notna().sum()
    print(f"       Expert 1 filled: {has_expert1} / {len(review)}")
    print(f"       Expert 2 filled: {has_expert2} / {len(review)}")
    print(f"       Consensus filled: {has_consensus} / {len(review)}")

    if has_consensus == 0 and has_expert1 == 0:
        print("\n  [warn] No assignments found in the returned workbook.")
        print("         Ensure experts have filled in the Expert/Consensus columns.")
        print("         Exiting without writing model spec.")
        return

    # ── Agreement metrics ──────────────────────────────────────────────────
    print("\n[3/5] Computing agreement metrics …")
    merged = (
        review
        .merge(step1, on="item_id", how="left")
        .merge(step3,  on="item_id", how="left")
    )

    results: dict[str, float | None] = {}

    if has_expert1 > 0 and has_expert2 > 0:
        k = _kappa(merged["expert1"].tolist(), merged["expert2"].tolist(),
                   "AD vs AH")
        results["AD vs AH (IRR)"] = k
        if k is not None:
            print(f"   AD vs AH (IRR)                       κ = {k:.4f}")

    if has_expert1 > 0:
        k = _kappa(merged["expert1"].tolist(), merged["step3_factor"].tolist(),
                   "AD vs Step3")
        results["AD vs Step 3 (NLP)"] = k
        if k is not None:
            print(f"   AD vs Step 3 (bottom-up NLP)         κ = {k:.4f}")

        k = _kappa(merged["expert1"].tolist(), merged["step1_factor"].tolist(),
                   "AD vs Step1")
        results["AD vs Step 1 (theoretical)"] = k
        if k is not None:
            print(f"   AD vs Step 1 (theoretical)           κ = {k:.4f}")

    if has_expert2 > 0:
        k = _kappa(merged["expert2"].tolist(), merged["step3_factor"].tolist(),
                   "AH vs Step3")
        results["AH vs Step 3 (NLP)"] = k
        if k is not None:
            print(f"   AH vs Step 3 (bottom-up NLP)         κ = {k:.4f}")

        k = _kappa(merged["expert2"].tolist(), merged["step1_factor"].tolist(),
                   "AH vs Step1")
        results["AH vs Step 1 (theoretical)"] = k
        if k is not None:
            print(f"   AH vs Step 1 (theoretical)           κ = {k:.4f}")

    if has_consensus > 0:
        k = _kappa(merged["consensus"].tolist(), merged["step3_factor"].tolist(),
                   "Consensus vs Step3")
        results["Consensus vs Step 3 (NLP)"] = k
        if k is not None:
            print(f"   Consensus vs Step 3 (bottom-up NLP)  κ = {k:.4f}")

        k = _kappa(merged["consensus"].tolist(), merged["step1_factor"].tolist(),
                   "Consensus vs Step1")
        results["Consensus vs Step 1 (theoretical)"] = k
        if k is not None:
            print(f"   Consensus vs Step 1 (theoretical)    κ = {k:.4f}")

    # ── Confusion matrices & structural swap detection ─────────────────────
    rater = "consensus" if has_consensus > 0 else "expert1"
    print(f"\n   Confusion analysis using '{rater}' as reference rater.")

    cm_vs_step1 = _confusion(merged[rater].tolist(), merged["step1_factor"].tolist())
    cm_vs_step3 = _confusion(merged[rater].tolist(), merged["step3_factor"].tolist())

    if not cm_vs_step1.empty:
        print_confusion(cm_vs_step1, f"{rater.title()} (rows) vs Step 1 Theoretical (cols)")
        swaps1 = _detect_structural_swaps(cm_vs_step1)
        if swaps1:
            print(f"\n   Structural label-swaps detected vs Step 1:")
            for a, b in swaps1:
                print(f"     {a} → {b}  (≥70 % of items redirected)")

    if not cm_vs_step3.empty:
        print_confusion(cm_vs_step3, f"{rater.title()} (rows) vs Step 3 NLP (cols)")
        swaps3 = _detect_structural_swaps(cm_vs_step3)
        if swaps3:
            print(f"\n   Structural label-swaps detected vs Step 3:")
            for a, b in swaps3:
                print(f"     {a} → {b}  (≥70 % of items redirected)")

    # ── Persist agreement metrics for the manuscript ───────────────────────
    kappa_out = _REPO / "expert_review/records/coauthor_review_b3_kappa.json"
    kappa_csv = _REPO / "expert_review/records/coauthor_review_b3_kappa.csv"
    import json as _json

    # ── The same metrics on the items that are discordant under the FINAL
    # translated structure.  The review set (61 items) is the union of the
    # discordant sheet and the six UC-flagged items, defined before the
    # experts' consensus overrode three UC assignments.  "final" applies the
    # consensus to the UC-flagged items (exactly what
    # analysis/item_assignments.py does), so the subset below equals
    # results/tables/mismatches_ledger_b3.csv (60 items: COG_027 was reviewed
    # as a UC item but is concordant; COG_033 became discordant through its
    # override).
    merged["final_factor"] = [
        c if (uc and not pd.isna(c)) else s1
        for uc, c, s1 in zip(merged["is_uc"], merged["consensus"], merged["step1_factor"])
    ]
    final_disc = merged[merged["final_factor"] != merged["step3_factor"]]
    sub_results: dict[str, float | None] = {}
    for label, a, b in [("AD vs AH (IRR)", "expert1", "expert2"),
                        ("AD vs Step 1 (theoretical)", "expert1", "step1_factor"),
                        ("AH vs Step 1 (theoretical)", "expert2", "step1_factor"),
                        ("AD vs final translated", "expert1", "final_factor"),
                        ("AH vs final translated", "expert2", "final_factor"),
                        ("AD vs Step 3 (NLP)", "expert1", "step3_factor"),
                        ("AH vs Step 3 (NLP)", "expert2", "step3_factor"),
                        ("Consensus vs Step 1 (theoretical)", "consensus", "step1_factor"),
                        ("Consensus vs final translated", "consensus", "final_factor"),
                        ("Consensus vs Step 3 (NLP)", "consensus", "step3_factor")]:
        sub_results[label] = _kappa(final_disc[a].tolist(), final_disc[b].tolist(), label)
    print(f"\n   Discordant under the final translated structure: {len(final_disc)} of "
          f"{len(merged)} reviewed items (reviewed but concordant: "
          f"{sorted(set(merged['item_id']) - set(final_disc['item_id']))})")
    for k, v in sub_results.items():
        if v is not None:
            print(f"   [final-discordant] {k:<36s} κ = {v:.4f}")
    # the two Step-1-vs-final comparisons on the full review set as well
    for label, a in (("AD vs final translated", "expert1"), ("AH vs final translated", "expert2"),
                     ("Consensus vs final translated", "consensus")):
        results[label] = _kappa(merged[a].tolist(), merged["final_factor"].tolist(), label)

    payload = {
        "input": str(input_path.name),
        "n_reviewed": int(len(review)),
        "n_discordant_sheet": int((~review["is_uc"]).sum()),
        "n_uc_flagged": int(review["is_uc"].sum()),
        "items_reviewed": sorted(review["item_id"].tolist()),
        "n_expert1": int(has_expert1),
        "n_expert2": int(has_expert2),
        "n_consensus": int(has_consensus),
        "kappa": {k: (None if v is None else round(float(v), 4))
                  for k, v in results.items()},
        "final_discordant_subset": {
            "n_items": int(len(final_disc)),
            "reviewed_but_concordant": sorted(set(merged["item_id"]) - set(final_disc["item_id"])),
            "n_expert1": int(final_disc["expert1"].notna().sum()),
            "n_expert2": int(final_disc["expert2"].notna().sum()),
            "n_consensus": int(final_disc["consensus"].notna().sum()),
            "kappa": {k: (None if v is None else round(float(v), 4))
                      for k, v in sub_results.items()},
        },
    }
    if dry_run:
        print(f"\n   [dry-run] Agreement metrics (not written):\n   "
              + _json.dumps(payload, ensure_ascii=False))
    else:
        with open(kappa_out, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2, ensure_ascii=False)
        pd.DataFrame(
            [{"comparison": k, "kappa": v} for k, v in results.items()]
        ).to_csv(kappa_csv, index=False)
        print(f"\n   Agreement metrics written to:\n     {kappa_out}\n     {kappa_csv}")

    # ── Build full 91-item model ───────────────────────────────────────────
    print("\n[4/5] Building full 91-item assignment …")
    full = step1.copy()

    # Determine which column to use for reviewed items
    use_col = "consensus" if has_consensus > 0 else "expert1"

    # MODEL IS FROZEN: only the Step-1 UC-ambiguous items may override the
    # theoretical assignment. The ~60 discordant items are validation-only
    # (they still feed the kappa/confusion analysis above) and must NOT reshape
    # the measurement model that Results/Discussion are already written on.
    uc_mask = review["is_uc"] if "is_uc" in review.columns \
        else review["sheet"].apply(_is_uc_sheet)
    uc_review = review[uc_mask]
    n_uc_total = len(uc_review)
    review_map = (
        uc_review[uc_review[use_col].notna()][["item_id", use_col]]
        .set_index("item_id")[use_col]
        .to_dict()
    )
    n_discordant_ignored = int((~uc_mask & review[use_col].notna()).sum())
    print(f"   Model-building restricted to UC sheet: {n_uc_total} UC items "
          f"({len(review_map)} with a usable '{use_col}' label).")
    print(f"   Discordant items used for validation only (not in model): "
          f"{n_discordant_ignored}.")

    full["final_factor"] = full.apply(
        lambda r: review_map.get(r["item_id"], r["step1_factor"]),
        axis=1,
    )
    full["source"] = full["item_id"].apply(
        lambda iid: "coauthor_consensus" if iid in review_map else "step1_theoretical"
    )

    n_consensus_used = sum(1 for v in full["source"] if v == "coauthor_consensus")
    n_step1_fallback = sum(1 for v in full["source"] if v == "step1_theoretical")

    print(f"   Items from coauthor consensus : {n_consensus_used}")
    print(f"   Items from Step 1 fallback    : {n_step1_fallback}")
    print(f"   Total                         : {len(full)}")

    # Factor distribution summary
    dist = full["final_factor"].value_counts()
    print("\n   Final factor distribution:")
    for fac in FACTOR_ORDER:
        n = dist.get(fac, 0)
        print(f"     {fac:<40s} {n:>3}")

    # ── Save model spec ────────────────────────────────────────────────────
    print(f"\n[5/5] Writing model spec '{model_name}' …")
    spec = build_model_spec(
        full, model_name,
        n_reviewed=len(review),
        n_consensus_used=n_consensus_used,
        n_step1_fallback=n_step1_fallback,
    )
    save_model_spec(spec, model_name, dry_run)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"\n  Model spec '{model_name}' is the final 91-item taxonomy used by the paper;")
    print("  build the tabular version with:  python analysis/item_assignments.py")
    print("  (the empirical CFA/MIMIC models in psychometrics/ use the 23 items 34-68 retained")
    print("   after floor/ceiling screening; the expert overrides lie outside that set)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import co-author review and compute Cohen's kappa for B3 items."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Returned review Excel workbook (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--model-name", "-m",
        default=DEFAULT_MODEL,
        help=f"Name for the new model spec entry (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print spec to stdout instead of writing to model_specs.yaml",
    )
    parser.add_argument(
        "--from-per-expert",
        action="store_true",
        help="Force reading the per-expert workbooks (A.D./A.H.) instead of the "
             "merged solution workbook. Auto-enabled when the solution file is blank.",
    )
    parser.add_argument(
        "--expert1-file",
        type=Path, default=None,
        help="Expert 1 (A.D.) per-reviewer workbook "
             "(default: expert_review/records/coauthor_review_b3_AD.xlsx)",
    )
    parser.add_argument(
        "--expert2-file",
        type=Path, default=None,
        help="Expert 2 (A.H.) per-reviewer workbook "
             "(default: expert_review/records/coauthor_review_b3_AH.xlsx)",
    )
    args = parser.parse_args()
    main(input_path=args.input, model_name=args.model_name, dry_run=args.dry_run,
         from_per_expert=args.from_per_expert,
         expert1_file=args.expert1_file, expert2_file=args.expert2_file)
