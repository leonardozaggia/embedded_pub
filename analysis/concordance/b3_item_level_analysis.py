
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Item-Level Assignment Analysis — B3 Edition.
Compares theoretical (step1_translation) vs detected (step3_b3_bottomup).

Outputs:
  1. item_assignments_b3.xlsx  — Excel workbook
  2. item_alluvial_b3.png      — 2-column alluvial figure (Theoretical | Detected)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bayley_nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import update_results_json, ensure_dir, export_mismatches_ledger


FACTOR_COLORS = {
    "WM":   "#4C72B0",
    "FS":   "#DD8452",
    "GDPS": "#55A868",
    "ATT":  "#C44E52",
    "HOP":  "#8172B3",
}
UNASSIGNED_COLOR = "#CCCCCC"

THEO_ABBREV = {
    "attention": "ATT",
    "working_memory": "WM",
    "goal_directed_problem_solving": "GDPS",
    "flexibility_shift": "FS",
    "higher_order_processing": "HOP",
}
CANONICAL_THEO_ORDER = ["WM", "FS", "GDPS", "ATT", "HOP"]

DETECTED_ABBREV_MAP = {
    "working memory": "WM",
    "attention": "ATT",
    "higher order processing": "HOP",
    "higher order processes": "HOP",
    "flexibility shift": "FS",
    "goal directed problem solving": "GDPS",
}


def abbrev_detected(label: str) -> str:
    key = str(label).strip().lower().replace("_", " ").replace("-", " ")
    return DETECTED_ABBREV_MAP.get(key, label[:8].upper())


def format_scoring(scoring_json_str) -> str:
    try:
        entries = json.loads(str(scoring_json_str))
        parts = []
        for e in entries:
            pts = e.get("points", "?")
            crit = str(e.get("criteria", "")).rstrip(" -")
            parts.append(f"{pts}pt: {crit}")
        return "  |  ".join(parts)
    except Exception:
        return str(scoring_json_str)


def load_cluster_labels(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path)
    if "cluster_id" in df.columns and "top_concept" in df.columns:
        return {int(r["cluster_id"]): str(r["top_concept"]) for _, r in df.iterrows()}
    raise ValueError("Unexpected format in cluster_concept_labels.csv")


def load_theoretical_from_b3_csv(assignments_csv: Path) -> Dict[str, List[str]]:
    """Load the translated (Objective 1) factor assignments for all 91 items.

    Reads results/tables/bayley3_item_domain_assignments.csv (cascade pairing +
    centroid assignment + expert review) via analysis/concordance/_translated.py.
    """
    from _translated import load_theoretical_dict
    return load_theoretical_dict(assignments_csv.parents[2])


def build_master_df(
    items_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_label_map: Dict[int, str],
    theo_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    item_to_theo: Dict[str, List[str]] = {}
    for factor_key, item_ids in theo_dict.items():
        abbr = THEO_ABBREV.get(factor_key, factor_key.upper())
        for iid in item_ids:
            item_to_theo.setdefault(iid, []).append(abbr)

    cluster_lookup = cluster_df.set_index("item_id")

    rows = []
    for _, row in items_df.iterrows():
        iid = row["item_id"]
        title = row.get("item_title", "")
        description = str(row.get("item_description", "") or "")
        scoring = format_scoring(row["scoring_json"]) if "scoring_json" in row.index and pd.notna(row.get("scoring_json")) else ""

        if iid in cluster_lookup.index:
            det_num = int(cluster_lookup.loc[iid, "cluster"])
            det_prob = float(cluster_lookup.loc[iid, "probability"]) if "probability" in cluster_lookup.columns else np.nan
            det_label_raw = cluster_label_map.get(det_num, f"C{det_num}")
            det_abbrev = abbrev_detected(det_label_raw)
        else:
            det_num, det_prob, det_label_raw, det_abbrev = np.nan, np.nan, "—", "—"

        theo_list = item_to_theo.get(iid, [])

        rows.append({
            "item_id": iid,
            "item_title": title,
            "item_description": description,
            "scoring_summary": scoring,
            "detected_cluster_num": det_num,
            "detected_cluster_label": det_label_raw,
            "detected_abbrev": det_abbrev,
            "detected_probability": round(det_prob, 3) if not np.isnan(det_prob) else "",
            "theoretical_factors": ", ".join(theo_list) if theo_list else "—",
            "theo_det_agree": int(bool(theo_list and det_abbrev in theo_list)),
        })

    df = pd.DataFrame(rows)

    def sort_key(row):
        theo = row["theoretical_factors"].split(", ")[0] if row["theoretical_factors"] != "—" else "ZZZ"
        order = CANONICAL_THEO_ORDER.index(theo) if theo in CANONICAL_THEO_ORDER else 99
        return (order, row["item_id"])

    df = df.loc[sorted(df.index, key=lambda i: sort_key(df.loc[i]))]
    df = df.reset_index(drop=True)
    return df


def _hex_to_openpyxl_fill(hex_color: str) -> PatternFill:
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = int(r + (255 - r) * 0.55)
    g = int(g + (255 - g) * 0.55)
    b = int(b + (255 - b) * 0.55)
    argb = f"FF{r:02X}{g:02X}{b:02X}"
    return PatternFill(start_color=argb, end_color=argb, fill_type="solid")


def write_excel(master_df: pd.DataFrame, output_path: Path) -> None:
    thin = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def fill_for(abbrev: str) -> Optional[PatternFill]:
        color = FACTOR_COLORS.get(abbrev)
        return _hex_to_openpyxl_fill(color) if color else None

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        master_df.to_excel(writer, sheet_name="Master", index=False)
        ws = writer.sheets["Master"]

        header_fill = PatternFill(start_color="FF2E4057", end_color="FF2E4057", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFFFF", size=11)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = border

        col_widths = {
            "item_id": 12, "item_title": 38,
            "item_description": 55, "scoring_summary": 70,
            "detected_cluster_num": 8,
            "detected_cluster_label": 22, "detected_abbrev": 10,
            "detected_probability": 10, "theoretical_factors": 24,
            "theo_det_agree": 10,
        }
        for col_idx, col_name in enumerate(master_df.columns, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 14)

        det_col_idx = list(master_df.columns).index("detected_abbrev") + 1
        agree_col_idx = list(master_df.columns).index("theo_det_agree") + 1
        desc_col_idx = list(master_df.columns).index("item_description") + 1
        score_col_idx = list(master_df.columns).index("scoring_summary") + 1
        n_cols = len(master_df.columns)

        for row_idx in range(2, len(master_df) + 2):
            det_abbrev = ws.cell(row=row_idx, column=det_col_idx).value
            row_fill = fill_for(str(det_abbrev)) if det_abbrev else None

            agree_val = ws.cell(row=row_idx, column=agree_col_idx).value
            for col_idx in range(1, n_cols + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = border
                if col_idx in (desc_col_idx, score_col_idx):
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    cell.alignment = Alignment(wrap_text=False, vertical="center")
                if row_fill:
                    cell.fill = row_fill

            agree_cell = ws.cell(row=row_idx, column=agree_col_idx)
            if agree_val == 0:
                agree_cell.font = Font(bold=True, color="FFCC0000")
            else:
                agree_cell.font = Font(bold=True, color="FF006400")

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Per-theoretical-factor sheets
        for factor_abbrev in CANONICAL_THEO_ORDER:
            sub = master_df[
                master_df["theoretical_factors"].str.contains(factor_abbrev, na=False)
            ].copy()
            if sub.empty:
                continue
            sub.to_excel(writer, sheet_name=f"Theo_{factor_abbrev}", index=False)
            ws2 = writer.sheets[f"Theo_{factor_abbrev}"]
            for cell in ws2[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")
                cell.border = border
            ws2.column_dimensions["A"].width = 12
            ws2.column_dimensions["B"].width = 38
            for col_idx in range(3, len(sub.columns) + 1):
                ws2.column_dimensions[get_column_letter(col_idx)].width = 14
            ws2.freeze_panes = "A2"

        # Disagreements sheet
        disagree = master_df[master_df["theo_det_agree"] == 0].copy()
        disagree.to_excel(writer, sheet_name="Disagreements", index=False)
        ws3 = writer.sheets["Disagreements"]
        for cell in ws3[1]:
            cell.fill = PatternFill(start_color="FFCC0000", end_color="FFCC0000", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFFFF", size=11)
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        ws3.column_dimensions["A"].width = 12
        ws3.column_dimensions["B"].width = 38
        ws3.freeze_panes = "A2"

        # Cross-tab sheet
        pivot = pd.crosstab(
            master_df["theoretical_factors"],
            master_df["detected_abbrev"],
            margins=True,
            margins_name="Total",
        )
        pivot.to_excel(writer, sheet_name="CrossTab_Theo_x_Det")

        # Compact sheet
        compact = master_df[
            ["item_id", "item_title", "item_description", "scoring_summary",
             "theoretical_factors", "detected_abbrev", "theo_det_agree"]
        ].copy()
        compact.to_excel(writer, sheet_name="Compact", index=False)
        ws5 = writer.sheets["Compact"]
        for cell in ws5[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        ws5.column_dimensions["A"].width = 12
        ws5.column_dimensions["B"].width = 38
        ws5.column_dimensions["C"].width = 55
        ws5.column_dimensions["D"].width = 70
        ws5.column_dimensions["E"].width = 24
        ws5.column_dimensions["F"].width = 14
        ws5.column_dimensions["G"].width = 10
        ws5.freeze_panes = "A2"

    print(f"  Excel saved -> {output_path}")


def _factor_color(label: str) -> str:
    for k, v in FACTOR_COLORS.items():
        if k in str(label).upper():
            return v
    return UNASSIGNED_COLOR


def _draw_bezier(ax, x0, y0, x1, y1, color, alpha=0.35, lw=1.0):
    cx = (x0 + x1) / 2
    verts = [(x0, y0), (cx, y0), (cx, y1), (x1, y1)]
    codes = [1, 4, 4, 4]
    from matplotlib.path import Path as MPath
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        alpha=alpha,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)


def plot_alluvial(master_df: pd.DataFrame, output_path: Path) -> None:
    """Two vertical axes: Theoretical | Detected."""
    all_theo = []
    for abbrev in CANONICAL_THEO_ORDER:
        if master_df["theoretical_factors"].str.contains(abbrev, na=False).any():
            all_theo.append(abbrev)

    all_det_raw = master_df["detected_abbrev"].dropna().unique().tolist()
    det_order_map = {v: i for i, v in enumerate(CANONICAL_THEO_ORDER)}
    all_det = sorted(all_det_raw, key=lambda x: (det_order_map.get(x, 99), x))

    ITEM_H = 0.55
    BAND_GAP = 1.2
    AXIS_X = {0: 0.0, 1: 6.0}
    LABEL_OFFSET_LEFT = -0.22
    FONT_SIZE_ITEM = 10.0
    FONT_SIZE_FACTOR = 14.0

    def compute_band_positions(factor_list, item_counts):
        result = {}
        cursor = 0.0
        for fac in reversed(factor_list):
            n = item_counts.get(fac, 0)
            height = max(n * ITEM_H, ITEM_H)
            y_start = cursor
            y_end = cursor + height
            ys = [cursor + (k + 0.5) * ITEM_H for k in range(n)]
            result[fac] = (y_start, y_end, ys)
            cursor = y_end + BAND_GAP
        return result, cursor

    theo_items: Dict[str, List[int]] = {f: [] for f in all_theo}
    for idx, row in master_df.iterrows():
        factors = [t.strip() for t in row["theoretical_factors"].split(", ") if t.strip() != "—"]
        primary = factors[0] if factors else None
        if primary and primary in theo_items:
            theo_items[primary].append(idx)

    theo_counts = {f: len(v) for f, v in theo_items.items()}
    det_counts = master_df["detected_abbrev"].value_counts().to_dict()

    theo_bands, total_height_theo = compute_band_positions(all_theo, theo_counts)
    det_bands, total_height_det = compute_band_positions(all_det, det_counts)

    total_height = max(total_height_theo, total_height_det)

    FIG_W = 20
    FIG_H = max(16, total_height * 0.38)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-3.5, 9.0)
    ax.set_ylim(-1, total_height + 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    BAND_W = 0.35

    def draw_bands(bands, factor_list, x_col, side="left"):
        for fac in factor_list:
            if fac not in bands:
                continue
            y_start, y_end, _ = bands[fac]
            color = FACTOR_COLORS.get(fac, UNASSIGNED_COLOR)
            rect = mpatches.FancyBboxPatch(
                (x_col - BAND_W / 2, y_start),
                BAND_W, y_end - y_start,
                boxstyle="round,pad=0.05",
                linewidth=0.8,
                edgecolor="white",
                facecolor=color,
                zorder=3,
            )
            ax.add_patch(rect)
            label_x = x_col + BAND_W / 2 + 0.15 if side == "right" else x_col - BAND_W / 2 - 0.15
            ha = "left" if side == "right" else "right"
            ax.text(
                label_x, (y_start + y_end) / 2, fac,
                ha=ha, va="center",
                fontsize=FONT_SIZE_FACTOR,
                fontweight="bold",
                color=color,
                zorder=4,
            )

    draw_bands(theo_bands, all_theo, AXIS_X[0], side="right")
    draw_bands(det_bands, all_det, AXIS_X[1], side="right")

    for x_col, title in zip(AXIS_X.values(), ["Theoretical", "Detected"]):
        ax.text(x_col, total_height + 0.4, title, ha="center", va="bottom",
                fontsize=18, fontweight="bold", color="#222222")

    det_slot: Dict[str, int] = {f: 0 for f in all_det}
    item_y_det: Dict[int, float] = {}

    for idx, row in master_df.iterrows():
        det = row["detected_abbrev"]
        if det in det_slot and det in det_bands:
            slot = det_slot[det]
            ys = det_bands[det][2]
            if slot < len(ys):
                item_y_det[idx] = ys[slot]
            det_slot[det] += 1

    drawn_labels: set = set()

    for idx, row in master_df.iterrows():
        iid = row["item_id"]
        title = str(row.get("item_title", ""))[:42]
        det = row["detected_abbrev"]
        theo_factors = [t.strip() for t in row["theoretical_factors"].split(", ") if t.strip() != "—"]
        primary_theo = theo_factors[0] if theo_factors else None

        if primary_theo not in theo_bands:
            continue

        theo_ys = theo_bands[primary_theo][2]
        theo_slot_idx = theo_items[primary_theo].index(idx)
        y_theo = theo_ys[theo_slot_idx] if theo_slot_idx < len(theo_ys) else theo_bands[primary_theo][2][-1]

        y_det = item_y_det.get(idx, None)
        color_theo = FACTOR_COLORS.get(primary_theo, UNASSIGNED_COLOR)

        if y_det is not None:
            _draw_bezier(ax, AXIS_X[0] + BAND_W / 2, y_theo,
                         AXIS_X[1] - BAND_W / 2, y_det,
                         color_theo, alpha=0.30, lw=0.9)

        if iid not in drawn_labels:
            label_text = f"{iid}  {title}"
            ax.text(
                AXIS_X[0] + LABEL_OFFSET_LEFT - BAND_W / 2,
                y_theo,
                label_text,
                ha="right", va="center",
                fontsize=FONT_SIZE_ITEM,
                color="#333333",
                fontfamily="monospace",
                zorder=5,
            )
            drawn_labels.add(iid)

    handles = [
        mpatches.Patch(color=color, label=abbrev)
        for abbrev, color in FACTOR_COLORS.items()
        if any(abbrev in str(c) for c in (
            master_df["theoretical_factors"].tolist()
            + master_df["detected_abbrev"].tolist()
        ))
    ]
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              frameon=True, fontsize=13, title="Factor", title_fontsize=13,
              bbox_to_anchor=(0.5, -0.02), bbox_transform=ax.transAxes)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Alluvial figure saved -> {output_path}")


def resolve_paths(args, root):
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else root / "data" / "outputs" / "step3_b3_bottomup" / "all_mpnet_base_v2_kmeans_consensus_no_dr"
    )
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "outputs" / "concordance" / "b3"

    assignments_csv = root / "results" / "tables" / "bayley3_item_domain_assignments.csv"

    return run_dir, assignments_csv, output_dir


def main():
    parser = argparse.ArgumentParser(description="Item-level assignment analysis — B3")
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir, assignments_csv, out_dir = resolve_paths(args, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    items_df = pd.read_csv(run_dir / "items.csv")
    cluster_df = pd.read_csv(run_dir / "03_clustering" / "cluster_assignments.csv")
    cluster_label_map = load_cluster_labels(run_dir / "04_labeling" / "cluster_concept_labels.csv")
    theo_dict = load_theoretical_from_b3_csv(assignments_csv)

    print("Building master table…")
    master_df = build_master_df(items_df, cluster_df, cluster_label_map, theo_dict)

    print("Writing Excel workbook…")
    write_excel(master_df, out_dir / "item_assignments_b3.xlsx")

    print("Rendering alluvial figure…")
    (root / "results" / "figures" / "supplementary").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    plot_alluvial(master_df, root / "results" / "figures" / "supplementary" / "figS4_b3_alluvial.png")

    n_agree = master_df["theo_det_agree"].sum()
    n_total = len(master_df)
    print(f"\nSummary: {n_agree}/{n_total} items where Detected agrees with Theoretical")
    print(f"Outputs in: {out_dir}")

    # ── Export structured metrics ────────────────────────────────────────────
    results_dir = root / "results" / "metrics"
    ensure_dir(str(results_dir))
    metrics_file = str(results_dir / "manuscript_metrics.json")

    n_agree_int = int(master_df["theo_det_agree"].sum())
    domain_acc = {}
    for factor in CANONICAL_THEO_ORDER:
        sub = master_df[master_df["theoretical_factors"].str.contains(factor, na=False)]
        if len(sub) > 0:
            n_f_agree = int(sub["theo_det_agree"].sum())
            domain_acc[factor] = {
                "n_items": len(sub),
                "n_agree": n_f_agree,
                "pct_agreement": round(100 * n_f_agree / len(sub), 1),
            }
    update_results_json("Bayley3_ItemLevel_Theo_vs_Det", {
        "overall_agreement_rate_pct": round(100 * n_agree_int / n_total, 1),
        "overall_agreement_fraction": f"{n_agree_int}/{n_total}",
        "domain_specific_accuracy": domain_acc,
    }, filename=metrics_file)

    export_mismatches_ledger(
        master_df, "theoretical_factors", "detected_abbrev", "item_id",
        filename=str(root / "results" / "tables" / "mismatches_ledger_b3.csv"),
    )


if __name__ == "__main__":
    main()
