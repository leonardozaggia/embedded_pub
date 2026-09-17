#!/usr/bin/env python
"""
analyze_paired_vs_extension_disagreement.py
============================================
Tests whether expert disagreement (on the B3 co-author review) is concentrated
in the 52 items ADDED to the 39-item cascade-paired "translated" model during
the Step-1 centroid-based extension (i.e. items whose factor label was never
directly inherited from an Aylward et al. (2022) Bayley-4 pairing).

step1_source categories capture HOW confidently the Step-1 pipeline placed each
item among the five cognitive factors — an interpretable, threshold-free proxy
for the algorithm's own certainty:
    cascade_pairing  -> item IS one of the 39 directly Bayley-4-paired items
                        (label inherited from Aylward et al. 2022 — highest
                        certainty; the "translated" core)
    centroid_loo     -> item added by centroid assignment, and one of the five
                        cognitive factors won the argmax cleanly (confident
                        extension)
    uc_redirected    -> item added by centroid assignment, but the item's single
                        nearest centroid was the UNCLASSIFIED cluster (UC); the
                        label had to be redirected to the best-matching factor
                        (lowest certainty — the algorithm's top choice was "none
                        of the five factors")

Binary grouping used for the coarse test:
    paired    = cascade_pairing                (n=39 in the full model)
    extension = centroid_loo + uc_redirected   (n=52 in the full model)

For every reviewed item we know: AD, AH, consensus, step1_factor (translated+
extended label), step3_factor (bottom-up NLP label). We compute, split by
paired/extension:
    - AD vs AH agreement (IRR)
    - Expert (pooled: consensus where available, else each individual expert
      call) vs Step 1  agreement
    - Expert (pooled) vs Step 3 agreement
and test the paired-vs-extension difference in disagreement rate with Fisher's
exact test (small-N safe) on top of reporting raw agreement % and Cohen's kappa
per subgroup.

NB: the argmax-UC criterion (n=23 of the 52 unpaired items) is BROADER than the
uc_margin > 0.10 rule that actually escalated items to expert review in Step 1
(only 6 items). We use argmax-UC here because it is a smoother certainty signal
with usable subgroup sizes (~15-20 reviewed) rather than n=6.

Outputs
-------
    expert_review/records/paired_vs_extension_disagreement.csv   (item-level table)
    expert_review/records/paired_vs_extension_summary.csv        (group-level summary)
    data/outputs/expert_review/paired_vs_extension_disagreement.png (diagnostic figure)
"""
from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")
del _s

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

_REPO = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_review_b3 as imp3  # noqa: E402

STEP1_DIR = _REPO / "data/outputs/step1_translation"
CENTROID_METHOD = "mixed_centroids"

OUT_CSV_ITEMS   = _REPO / "expert_review/records/paired_vs_extension_disagreement.csv"
OUT_CSV_SUMMARY = _REPO / "expert_review/records/paired_vs_extension_summary.csv"
OUT_FIG_DIR     = _REPO / "data/outputs/expert_review"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_COLORS = {
    "attention":                     "#E41A1C",
    "working_memory":                "#377EB8",
    "goal_directed_problem_solving": "#4DAF4A",
    "flexibility_shift":             "#FF7F00",
    "higher_order_processing":       "#984EA3",
}


# ---------------------------------------------------------------------------
# Step-1 provenance (paired vs extension) per item
# ---------------------------------------------------------------------------
def load_step1_source() -> pd.DataFrame:
    paired_csv   = STEP1_DIR / "b3_paired_assignments.csv"
    unpaired_csv = STEP1_DIR / CENTROID_METHOD / "all_unpaired_b3_assignments.csv"

    paired = pd.read_csv(paired_csv)[["item_id"]].copy()
    paired["step1_source"] = "cascade_pairing"
    paired["group"] = "paired"

    unpaired = pd.read_csv(unpaired_csv)[["item_id", "assigned_factor"]].copy()
    unpaired["step1_source"] = np.where(
        unpaired["assigned_factor"] == "uncategorized_cognition",
        "uc_redirected", "centroid_loo",
    )
    unpaired["group"] = "extension"
    unpaired = unpaired.drop(columns="assigned_factor")

    out = pd.concat([paired, unpaired], ignore_index=True)
    assert out["item_id"].is_unique, "duplicate item_id across paired/unpaired sets"
    assert len(out) == 91, f"expected 91 B3 items, got {len(out)}"
    return out


# ---------------------------------------------------------------------------
# Build item-level review table (reuses import_coauthor_review_b3 loaders)
# ---------------------------------------------------------------------------
def build_review_table() -> pd.DataFrame:
    items = imp3.load_items()
    step1 = imp3.load_step1_assignments()
    step3 = imp3.load_step3_assignments()
    source = load_step1_source()

    solution_path = _REPO / "expert_review/records/coauthor_review_b3_solution.xlsx"
    review = imp3.load_review_xlsx(solution_path, items)
    solution_empty = (review["expert1"].notna().sum() == 0
                       and review["consensus"].notna().sum() == 0)
    if solution_empty:
        review = imp3.load_review_per_expert(
            _REPO / "expert_review/records/coauthor_review_b3_AD.xlsx",
            _REPO / "expert_review/records/coauthor_review_b3_AH.xlsx",
            items,
        )

    merged = (
        review
        .merge(step1, on="item_id", how="left")
        .merge(step3, on="item_id", how="left")
        .merge(source, on="item_id", how="left")
        .merge(items[["item_id", "item_title"]], on="item_id", how="left")
    )
    missing_source = merged["group"].isna().sum()
    if missing_source:
        print(f"  [warn] {missing_source} reviewed items missing step1_source tag")

    return merged


# ---------------------------------------------------------------------------
# Agreement stats
# ---------------------------------------------------------------------------
def _agree_rate(a: pd.Series, b: pd.Series) -> tuple[int, int, float | None]:
    """(n_pairs, n_agree, rate) over rows where both a and b are non-null."""
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return 0, 0, None
    agree = int((a[mask] == b[mask]).sum())
    return n, agree, agree / n


def _kappa_safe(a: pd.Series, b: pd.Series) -> float | None:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return None
    aa, bb = a[mask].tolist(), b[mask].tolist()
    if len(set(aa) | set(bb)) < 2:
        return None
    try:
        return cohen_kappa_score(aa, bb)
    except ValueError:
        return None


COMPARISONS = [
    ("AD vs AH (IRR)",  "expert1",   "expert2"),
    ("AD vs Step 1 (translated)", "expert1", "step1_factor"),
    ("AH vs Step 1 (translated)", "expert2", "step1_factor"),
    ("Consensus vs Step 1 (translated)", "consensus", "step1_factor"),
    ("AD vs Step 3 (bottom-up)", "expert1", "step3_factor"),
    ("AH vs Step 3 (bottom-up)", "expert2", "step3_factor"),
    ("Consensus vs Step 3 (bottom-up)", "consensus", "step3_factor"),
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each comparison and each group (paired/extension/all), report
    n pairs, n agree, % agreement, Cohen's kappa.
    """
    rows = []
    for label, col_a, col_b in COMPARISONS:
        for grp_name, grp_df in [("paired", df[df["group"] == "paired"]),
                                   ("extension", df[df["group"] == "extension"]),
                                   ("all", df)]:
            n, agree, rate = _agree_rate(grp_df[col_a], grp_df[col_b])
            k = _kappa_safe(grp_df[col_a], grp_df[col_b])
            rows.append({
                "comparison": label, "group": grp_name,
                "n_pairs": n, "n_agree": agree,
                "pct_agree": None if rate is None else round(100 * rate, 1),
                "kappa": None if k is None else round(float(k), 4),
            })
    return pd.DataFrame(rows)


def summarize_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Same as summarize(), but split into the three step1_source categories
    instead of the binary paired/extension grouping — separates confidently
    extended items (centroid_loo) from items whose single nearest centroid was
    the unclassified cluster (uc_redirected)."""
    rows = []
    for label, col_a, col_b in COMPARISONS:
        for src in ("cascade_pairing", "centroid_loo", "uc_redirected"):
            grp_df = df[df["step1_source"] == src]
            n, agree, rate = _agree_rate(grp_df[col_a], grp_df[col_b])
            rows.append({
                "comparison": label, "step1_source": src,
                "n_pairs": n, "n_agree": agree,
                "pct_agree": None if rate is None else round(100 * rate, 1),
            })
    return pd.DataFrame(rows)


def fisher_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    2x2 Fisher exact test (agree/disagree x paired/extension) per comparison,
    pooling disagreement across the individual-expert comparisons (expert1,
    expert2) vs each reference (step1, step3) to keep cell counts usable.
    """
    comparisons = [
        ("AD vs AH (IRR)", "expert1", "expert2"),
        ("Expert(pooled) vs Step 1", None, "step1_factor"),
        ("Expert(pooled) vs Step 3", None, "step3_factor"),
    ]
    rows = []
    for label, col_a, col_b in comparisons:
        if col_a is not None:
            sub = df[["group", col_a, col_b]].dropna()
            agree = (sub[col_a] == sub[col_b])
        else:
            # pool expert1 and expert2 rows (each expert's individual call vs reference)
            parts = []
            for c in ("expert1", "expert2"):
                s = df[["group", c, col_b]].dropna()
                s = s.rename(columns={c: "val"})
                parts.append(s)
            sub = pd.concat(parts, ignore_index=True)
            agree = (sub["val"] == sub[col_b])

        table = pd.crosstab(sub["group"], agree)
        # ensure both columns (False/True) and both rows (paired/extension) exist,
        # in a fixed order — use reindex, NOT `table[[False, True]]` (pandas
        # treats a same-length list of bools as a row mask, not column labels).
        table = table.reindex(columns=[False, True], fill_value=0)
        table = table.reindex(index=["paired", "extension"], fill_value=0)

        odds, p = fisher_exact(table.values)
        rows.append({
            "comparison": label,
            "paired_disagree": int(table.loc["paired", False]),
            "paired_agree": int(table.loc["paired", True]),
            "extension_disagree": int(table.loc["extension", False]),
            "extension_agree": int(table.loc["extension", True]),
            "odds_ratio": round(odds, 3) if np.isfinite(odds) else None,
            "p_value": round(p, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
_FISHER_LOOKUP = {
    "AD vs AH (IRR)": "AD vs AH (IRR)",
    "AD vs Step 1 (translated)": "Expert(pooled) vs Step 1",
    "AH vs Step 1 (translated)": "Expert(pooled) vs Step 1",
    "AD vs Step 3 (bottom-up)": "Expert(pooled) vs Step 3",
    "AH vs Step 3 (bottom-up)": "Expert(pooled) vs Step 3",
}


def plot_summary(summary: pd.DataFrame, fisher: pd.DataFrame,
                  source_summary: pd.DataFrame, out_path: Path) -> None:
    plot_labels = [
        "AD vs AH (IRR)",
        "AD vs Step 1 (translated)",
        "AH vs Step 1 (translated)",
        "Consensus vs Step 1 (translated)",
        "AD vs Step 3 (bottom-up)",
        "AH vs Step 3 (bottom-up)",
        "Consensus vs Step 3 (bottom-up)",
    ]
    sub = summary[summary["comparison"].isin(plot_labels) & summary["group"].isin(["paired", "extension"])]
    p_lookup = dict(zip(fisher["comparison"], fisher["p_value"]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.6))

    # ── Panel A: binary paired vs extension ─────────────────────────────────
    y_pos = np.arange(len(plot_labels))
    bar_h = 0.36
    color_paired    = "#4DAF4A"
    color_extension = "#E41A1C"

    for i, label in enumerate(plot_labels):
        row_p = sub[(sub["comparison"] == label) & (sub["group"] == "paired")]
        row_e = sub[(sub["comparison"] == label) & (sub["group"] == "extension")]
        pct_p = row_p["pct_agree"].iloc[0] if len(row_p) and row_p["pct_agree"].iloc[0] is not None else 0
        pct_e = row_e["pct_agree"].iloc[0] if len(row_e) and row_e["pct_agree"].iloc[0] is not None else 0
        n_p = row_p["n_pairs"].iloc[0] if len(row_p) else 0
        n_e = row_e["n_pairs"].iloc[0] if len(row_e) else 0

        axA.barh(i + bar_h / 2, pct_p, height=bar_h, color=color_paired,
                  label="Paired (translated core)" if i == 0 else None)
        axA.barh(i - bar_h / 2, pct_e, height=bar_h, color=color_extension,
                  label="Extension (centroid-added)" if i == 0 else None)

        axA.text(pct_p + 1.5, i + bar_h / 2, f"{pct_p:.0f}% (n={n_p})", va="center", fontsize=8.5)
        axA.text(pct_e + 1.5, i - bar_h / 2, f"{pct_e:.0f}% (n={n_e})", va="center", fontsize=8.5)

        fkey = _FISHER_LOOKUP.get(label)
        if fkey is not None:
            p = p_lookup.get(fkey)
            if p is not None:
                axA.text(101, i, f"p={p:.2f}", va="center", fontsize=7.5,
                          color="#555555", style="italic")

    axA.set_yticks(y_pos)
    axA.set_yticklabels(plot_labels, fontsize=9.5)
    axA.invert_yaxis()
    axA.set_xlabel("% agreement", fontsize=10)
    axA.set_xlim(0, 118)
    axA.axvline(50, color="grey", linestyle=":", linewidth=1, zorder=0)
    axA.set_title("(A) Paired (n=39) vs Extension (n=52) items\nFisher p from pooled 2x2 test", fontsize=11)
    axA.legend(loc="lower right", fontsize=8.5, frameon=False)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)

    # ── Panel B: three-way provenance breakdown ─────────────────────────────
    src_labels = [
        "AD vs AH (IRR)",
        "AD vs Step 1 (translated)",
        "AH vs Step 1 (translated)",
        "AD vs Step 3 (bottom-up)",
        "AH vs Step 3 (bottom-up)",
    ]
    src_colors = {
        "cascade_pairing": "#4DAF4A",
        "centroid_loo":    "#377EB8",
        "uc_redirected":   "#E41A1C",
    }
    src_display = {
        "cascade_pairing": "Paired (cascade)",
        "centroid_loo":    "Extension: confident (factor won argmax)",
        "uc_redirected":   "Extension: UC-redirected (unclassified won argmax)",
    }
    y_pos_b = np.arange(len(src_labels))
    bar_h3 = 0.24
    offsets = {"cascade_pairing": bar_h3, "centroid_loo": 0, "uc_redirected": -bar_h3}

    for i, label in enumerate(src_labels):
        for src, off in offsets.items():
            row = source_summary[(source_summary["comparison"] == label) &
                                  (source_summary["step1_source"] == src)]
            pct = row["pct_agree"].iloc[0] if len(row) and row["pct_agree"].iloc[0] is not None else 0
            n = row["n_pairs"].iloc[0] if len(row) else 0
            axB.barh(i + off, pct, height=bar_h3, color=src_colors[src],
                      label=src_display[src] if i == 0 else None)
            axB.text(pct + 1.5, i + off, f"{pct:.0f}% (n={n})", va="center", fontsize=7.5)

    axB.set_yticks(y_pos_b)
    axB.set_yticklabels(src_labels, fontsize=9.5)
    axB.invert_yaxis()
    axB.set_xlabel("% agreement", fontsize=10)
    axB.set_xlim(0, 100)
    axB.axvline(50, color="grey", linestyle=":", linewidth=1, zorder=0)
    axB.set_title("(B) Provenance breakdown: confident extension items\nvs items Step 1 itself already flagged uncertain (UC)", fontsize=11)
    axB.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)

    fig.suptitle(
        "Is expert disagreement concentrated in the centroid-extended items?",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  PAIRED (translated core) vs EXTENSION (centroid-added) DISAGREEMENT")
    print("=" * 70)

    df = build_review_table()
    n_paired = (df["group"] == "paired").sum()
    n_extension = (df["group"] == "extension").sum()
    print(f"\n  Reviewed items: {len(df)}  (paired={n_paired}, extension={n_extension})")
    print(df["step1_source"].value_counts(dropna=False).to_string())

    df.to_csv(OUT_CSV_ITEMS, index=False)
    print(f"\n  Item-level table -> {OUT_CSV_ITEMS}")

    summary = summarize(df)
    summary.to_csv(OUT_CSV_SUMMARY, index=False)
    print(f"  Summary table    -> {OUT_CSV_SUMMARY}\n")
    print(summary.to_string(index=False))

    fisher = fisher_tests(df)
    print("\n  Fisher exact tests (paired vs extension disagreement rate):")
    print(fisher.to_string(index=False))

    source_summary = summarize_by_source(df)
    source_summary.to_csv(_REPO / "expert_review/records/paired_vs_extension_by_source.csv", index=False)
    print("\n  Three-way provenance breakdown (cascade_pairing / centroid_loo / uc_redirected):")
    print(source_summary.pivot(index="comparison", columns="step1_source", values="pct_agree").to_string())

    plot_summary(summary, fisher, source_summary, OUT_FIG_DIR / "paired_vs_extension_disagreement.png")


if __name__ == "__main__":
    main()
