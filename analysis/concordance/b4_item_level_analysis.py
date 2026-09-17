
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Item-Level Assignment Analysis
================================
Produces two outputs for studying item assignments across theoretical,
detected, and empirical solutions:

1. item_assignments.xlsx  — comprehensive Excel workbook (5 sheets)
2. item_alluvial.png      — large alluvial / parallel-coordinates figure
                            with every item labelled by ID + title
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
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bayley_nlp"))
from utils import update_results_json, ensure_dir

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ── canonical colours — must match b4_sankey.py build_palette + CANONICAL_THEORETICAL_ORDER
# palette = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",...]
# order   =   WM         FS        GDPS      ATT       HOP
FACTOR_COLORS = {
    "WM":   "#4C72B0",   # blue
    "FS":   "#DD8452",   # orange
    "GDPS": "#55A868",   # green
    "ATT":  "#C44E52",   # red
    "HOP":  "#8172B3",   # purple
    # empirical fallbacks
    "F1":   "#937860",
    "F2":   "#DA8BC3",
    "F3":   "#8C8C8C",
    "F4":   "#CCB974",
    "F5":   "#64B5CD",
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

EMP_ABBREV_MAP = {
    "factor_1": "F1",
    "factor_2": "F2",
    "factor_3": "F3",
    "factor_4": "F4",
    "factor_5": "F5",
}


def abbrev_detected(label: str) -> str:
    key = str(label).strip().lower().replace("_", " ").replace("-", " ")
    return DETECTED_ABBREV_MAP.get(key, label[:8].upper())


def format_scoring(scoring_json_str) -> str:
    """Convert scoring JSON to a compact human-readable string."""
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


def load_all_b4_items_from_txt(txt_path: Path) -> Dict[str, Dict]:
    """Parse b4_cog_items.txt → {COG_XXX: {item_title, item_description, scoring_summary}}."""
    content = txt_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in content.split("---") if b.strip()]
    result: Dict[str, Dict] = {}
    for block in blocks:
        lines = block.split("\n")
        first_line = lines[0].strip()
        m = re.match(r"Item\s+(\d+)\s+[\u2013\u2014-]\s+(.+)", first_line)
        if not m:
            continue
        num = int(m.group(1))
        title = m.group(2).strip()
        item_id = f"COG_{num:03d}"
        rest = [l.strip() for l in lines[1:] if l.strip()]
        scoring_lines = [l for l in rest if re.match(r"\d+\s+Points?:", l, re.IGNORECASE)]
        desc_lines = [l for l in rest if not re.match(r"\d+\s+Points?:", l, re.IGNORECASE)]
        result[item_id] = {
            "item_title": title,
            "item_description": " ".join(desc_lines),
            "scoring_summary": "  |  ".join(scoring_lines),
        }
    return result


# ── loaders ─────────────────────────────────────────────────────────────────
def load_reference_models(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cluster_labels(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path)
    if "cluster_id" in df.columns and "top_concept" in df.columns:
        return {int(r["cluster_id"]): str(r["top_concept"]) for _, r in df.iterrows()}
    raise ValueError("Unexpected format in cluster_concept_labels.csv")


# ── data assembly ────────────────────────────────────────────────────────────
def build_master_df(
    items_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_label_map: Dict[int, str],
    theo_dict: Dict[str, List[str]],
    emp_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """Return one row per item with all three assignments."""

    # item_id → list of theoretical factor abbreviations
    item_to_theo: Dict[str, List[str]] = {}
    for factor_key, item_ids in theo_dict.items():
        abbr = THEO_ABBREV.get(factor_key, factor_key.upper())
        for iid in item_ids:
            item_to_theo.setdefault(iid, []).append(abbr)

    # item_id → list of empirical factor labels
    item_to_emp: Dict[str, List[str]] = {}
    for factor_key, item_ids in emp_dict.items():
        abbr = EMP_ABBREV_MAP.get(factor_key, factor_key.upper())
        for iid in item_ids:
            item_to_emp.setdefault(iid, []).append(abbr)

    # detected cluster label per item
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
        emp_list = item_to_emp.get(iid, [])

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
            "empirical_factors": ", ".join(emp_list) if emp_list else "—",
            # agreement flags
            "theo_det_agree": int(bool(theo_list and det_abbrev in theo_list)),
            "emp_det_agree": int(bool(emp_list and det_abbrev in emp_list)),
            "theo_emp_overlap": int(bool(set(theo_list) & set(emp_list))),
        })

    df = pd.DataFrame(rows)
    # Sort by primary theoretical factor then item_id
    def sort_key(row):
        theo = row["theoretical_factors"].split(", ")[0] if row["theoretical_factors"] != "—" else "ZZZ"
        order = CANONICAL_THEO_ORDER.index(theo) if theo in CANONICAL_THEO_ORDER else 99
        return (order, row["item_id"])

    df = df.loc[sorted(df.index, key=lambda i: sort_key(df.loc[i]))]
    df = df.reset_index(drop=True)
    return df


# ── Excel workbook ───────────────────────────────────────────────────────────
def _hex_to_openpyxl_fill(hex_color: str) -> PatternFill:
    c = hex_color.lstrip("#")
    # pastel: blend 55% toward white
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = int(r + (255 - r) * 0.55)
    g = int(g + (255 - g) * 0.55)
    b = int(b + (255 - b) * 0.55)
    argb = f"FF{r:02X}{g:02X}{b:02X}"
    return PatternFill(start_color=argb, end_color=argb, fill_type="solid")


def write_excel(master_df: pd.DataFrame, output_path: Path, unmodelled_df: Optional[pd.DataFrame] = None) -> None:
    thin = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── colour mapping for detected abbrev ─────────────────────────────────
    def fill_for(abbrev: str) -> Optional[PatternFill]:
        color = FACTOR_COLORS.get(abbrev)
        return _hex_to_openpyxl_fill(color) if color else None

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        # ── Sheet 1: Master table ───────────────────────────────────────────
        master_df.to_excel(writer, sheet_name="Master", index=False)
        ws = writer.sheets["Master"]

        # Header styling
        header_fill = PatternFill(start_color="FF2E4057", end_color="FF2E4057", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFFFF", size=11)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = border

        # Column widths
        col_widths = {
            "item_id": 12, "item_title": 38,
            "item_description": 55, "scoring_summary": 70,
            "detected_cluster_num": 8,
            "detected_cluster_label": 22, "detected_abbrev": 10,
            "detected_probability": 10, "theoretical_factors": 24,
            "empirical_factors": 20, "theo_det_agree": 10,
            "emp_det_agree": 10, "theo_emp_overlap": 12,
        }
        for col_idx, col_name in enumerate(master_df.columns, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 14)

        # Row data styling
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
                # wrap long text columns
                if col_idx in (desc_col_idx, score_col_idx):
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    cell.alignment = Alignment(wrap_text=False, vertical="center")
                if row_fill:
                    cell.fill = row_fill

            # Bold red for disagreements
            agree_cell = ws.cell(row=row_idx, column=agree_col_idx)
            if agree_val == 0:
                agree_cell.font = Font(bold=True, color="FFCC0000")
            else:
                agree_cell.font = Font(bold=True, color="FF006400")

        # ── Unmodelled items at the bottom (no colour) ──────────────────────
        if unmodelled_df is not None and not unmodelled_df.empty:
            # separator row
            sep_row = len(master_df) + 2
            ws.cell(row=sep_row, column=1).value = "── Items NOT included in the model ──"
            sep_fill = PatternFill(start_color="FFE0E0E0", end_color="FFE0E0E0", fill_type="solid")
            sep_font = Font(bold=True, italic=True, color="FF555555", size=10)
            for col_idx in range(1, n_cols + 1):
                c = ws.cell(row=sep_row, column=col_idx)
                c.fill = sep_fill
                c.font = sep_font
                c.border = border

            # write unmodelled rows using only columns that exist in master_df
            for rel_idx, (_, urow) in enumerate(unmodelled_df.iterrows()):
                data_row = sep_row + 1 + rel_idx
                for col_idx, col_name in enumerate(master_df.columns, start=1):
                    val = urow.get(col_name, "")
                    cell = ws.cell(row=data_row, column=col_idx, value=("" if pd.isna(val) else val))
                    cell.border = border
                    cell.font = Font(color="FF888888", italic=True)
                    if col_idx in (desc_col_idx, score_col_idx):
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                    else:
                        cell.alignment = Alignment(wrap_text=False, vertical="center")

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # ── Sheet 2: Per Theoretical Factor ────────────────────────────────
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

        # ── Sheet 3: Disagreements (theo ≠ detected) ───────────────────────
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

        # ── Sheet 4: Assignment Summary (cross-tab) ─────────────────────────
        pivot = pd.crosstab(
            master_df["theoretical_factors"],
            master_df["detected_abbrev"],
            margins=True,
            margins_name="Total",
        )
        pivot.to_excel(writer, sheet_name="CrossTab_Theo_x_Det")

        # ── Sheet 5: All three columns side-by-side compact ────────────────
        compact = master_df[
            ["item_id", "item_title", "item_description", "scoring_summary",
             "theoretical_factors", "detected_abbrev", "empirical_factors",
             "theo_det_agree", "emp_det_agree"]
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
        ws5.column_dimensions["G"].width = 20
        ws5.column_dimensions["H"].width = 10
        ws5.column_dimensions["I"].width = 10
        ws5.freeze_panes = "A2"

    print(f"  Excel saved → {output_path}")


# ── Alluvial / parallel-coordinates figure ───────────────────────────────────
def _factor_color(label: str) -> str:
    for k, v in FACTOR_COLORS.items():
        if k in str(label).upper():
            return v
    return UNASSIGNED_COLOR


def plot_alluvial(master_df: pd.DataFrame, output_path: Path) -> None:
    """
    Three vertical axes: Theoretical | Detected | Empirical
    Items are shown as labelled horizontal text on the left margin.
    Curved lines connect each item's assignment across columns.
    Nodes (factor bands) are drawn as solid coloured rectangles.
    """
    # ── collect all unique factor labels per column ──────────────────────────
    all_theo = []
    for abbrev in CANONICAL_THEO_ORDER:
        if master_df["theoretical_factors"].str.contains(abbrev, na=False).any():
            all_theo.append(abbrev)

    all_det_raw = master_df["detected_abbrev"].dropna().unique().tolist()
    # order detected by CANONICAL_THEO_ORDER where possible, then alphabetical
    det_order_map = {v: i for i, v in enumerate(CANONICAL_THEO_ORDER)}
    all_det = sorted(all_det_raw, key=lambda x: (det_order_map.get(x, 99), x))

    all_emp_raw = master_df["empirical_factors"].dropna().unique().tolist()
    all_emp = sorted([e for e in all_emp_raw if e != "—"], key=lambda x: x)

    # assign y-positions per factor in each column
    # Items are placed within factor bands; bands are evenly spaced.
    n_items = len(master_df)

    # --- layout constants ---
    # Sized for a 170 mm print width (Webappendix Figure S3): an 11-inch canvas
    # scales by ~0.6, so 10 pt item labels stay ~6 pt on the page.
    ITEM_H = 0.55          # vertical space per item (inches equivalent in data units)
    BAND_GAP = 1.2         # gap between factor bands
    AXIS_X = {0: 0.0, 1: 3.6, 2: 7.2}    # x positions of the three columns
    LABEL_OFFSET_LEFT = -0.22   # item label offset from axis 0
    FONT_SIZE_ITEM = 10.0
    FONT_SIZE_FACTOR = 13.0
    FONT_SIZE_TITLE = 15.0
    TITLE_CHARS = 36            # item titles truncated to keep the label column compact

    # ── compute y positions for each factor band in columns ─────────────────
    def compute_band_positions(factor_list, item_counts):
        """Return {factor: (y_start, y_end, [item_y_positions])} from bottom up."""
        result = {}
        cursor = 0.0
        for fac in reversed(factor_list):  # reversed so first factor is on top
            n = item_counts.get(fac, 0)
            height = max(n * ITEM_H, ITEM_H)
            y_start = cursor
            y_end = cursor + height
            ys = [cursor + (k + 0.5) * ITEM_H for k in range(n)]
            result[fac] = (y_start, y_end, ys)
            cursor = y_end + BAND_GAP
        return result, cursor

    # items sorted as in master_df (by primary theoretical factor)
    # Build per-factor item lists for left column
    theo_items: Dict[str, List[int]] = {f: [] for f in all_theo}
    for idx, row in master_df.iterrows():
        factors = [t.strip() for t in row["theoretical_factors"].split(", ") if t.strip() != "—"]
        primary = factors[0] if factors else None
        if primary and primary in theo_items:
            theo_items[primary].append(idx)

    theo_counts = {f: len(v) for f, v in theo_items.items()}
    det_counts = master_df["detected_abbrev"].value_counts().to_dict()
    emp_exploded = master_df["empirical_factors"].str.split(", ").explode()
    emp_counts = emp_exploded[emp_exploded != "—"].value_counts().to_dict()

    theo_bands, total_height_theo = compute_band_positions(all_theo, theo_counts)
    det_bands, total_height_det = compute_band_positions(all_det, det_counts)
    emp_bands, total_height_emp = compute_band_positions(all_emp, emp_counts)

    total_height = max(total_height_theo, total_height_det, total_height_emp)

    # ── figure setup ─────────────────────────────────────────────────────────
    FIG_W = 11
    FIG_H = max(13, total_height * 0.48)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-4.6, 8.7)
    ax.set_ylim(-1, total_height + 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── draw factor band rectangles ──────────────────────────────────────────
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
            # factor label
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
    draw_bands(emp_bands, all_emp, AXIS_X[2], side="right")

    # column titles
    for x_col, title in zip(AXIS_X.values(), ["Theoretical", "Detected", "Empirical"]):
        ax.text(x_col, total_height + 0.4, title, ha="center", va="bottom",
                fontsize=FONT_SIZE_TITLE, fontweight="bold", color="#222222")

    # ── assign item y-positions within their detected band ───────────────────
    # Track slot usage per (column, factor)
    det_slot: Dict[str, int] = {f: 0 for f in all_det}
    emp_slot: Dict[str, int] = {f: 0 for f in all_emp}

    item_y_det: Dict[int, float] = {}
    item_y_emp: Dict[int, Dict[str, float]] = {}

    # Pre-sort items by detected factor to fill bands top→bottom in order
    for idx, row in master_df.iterrows():
        det = row["detected_abbrev"]
        if det in det_slot and det in det_bands:
            slot = det_slot[det]
            ys = det_bands[det][2]
            if slot < len(ys):
                item_y_det[idx] = ys[slot]
            det_slot[det] += 1

    for idx, row in master_df.iterrows():
        emps = [e.strip() for e in row["empirical_factors"].split(", ") if e.strip() != "—"]
        item_y_emp[idx] = {}
        for e in emps:
            if e in emp_slot and e in emp_bands:
                slot = emp_slot[e]
                ys = emp_bands[e][2]
                if slot < len(ys):
                    item_y_emp[idx][e] = ys[slot]
                emp_slot[e] += 1

    # ── draw connecting lines & item labels ──────────────────────────────────
    drawn_labels: set = set()

    for idx, row in master_df.iterrows():
        iid = row["item_id"]
        title = str(row.get("item_title", ""))
        if len(title) > TITLE_CHARS:                  # truncate long titles
            title = title[:TITLE_CHARS - 1].rstrip() + "\u2026"
        det = row["detected_abbrev"]
        theo_factors = [t.strip() for t in row["theoretical_factors"].split(", ") if t.strip() != "—"]
        primary_theo = theo_factors[0] if theo_factors else None

        if primary_theo not in theo_bands:
            continue

        # y on theoretical axis: find position within factor band
        # Compute from band positions
        theo_ys = theo_bands[primary_theo][2]
        theo_slot_idx = theo_items[primary_theo].index(idx)
        y_theo = theo_ys[theo_slot_idx] if theo_slot_idx < len(theo_ys) else theo_bands[primary_theo][2][-1]

        y_det = item_y_det.get(idx, None)
        y_emp_dict = item_y_emp.get(idx, {})

        color_theo = FACTOR_COLORS.get(primary_theo, UNASSIGNED_COLOR)

        # Line: Theoretical → Detected
        if y_det is not None:
            _draw_bezier(ax, AXIS_X[0] + BAND_W / 2, y_theo,
                         AXIS_X[1] - BAND_W / 2, y_det,
                         color_theo, alpha=0.40, lw=1.3)

        # Line: Detected → Empirical
        for e_fac, y_emp in y_emp_dict.items():
            color_det = FACTOR_COLORS.get(det, UNASSIGNED_COLOR)
            _draw_bezier(ax, AXIS_X[1] + BAND_W / 2, y_det if y_det else y_theo,
                         AXIS_X[2] - BAND_W / 2, y_emp,
                         color_det, alpha=0.35, lw=1.3)

        # Item label on the left
        if iid not in drawn_labels:
            label_text = f"{iid}  {title}"
            ax.text(
                AXIS_X[0] + LABEL_OFFSET_LEFT - BAND_W / 2,
                y_theo,
                label_text,
                ha="right", va="center",
                fontsize=FONT_SIZE_ITEM,
                color="#333333",
                fontfamily="sans-serif",
                zorder=5,
            )
            drawn_labels.add(iid)

    # ── legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=color, label=abbrev)
        for abbrev, color in FACTOR_COLORS.items()
        if any(abbrev in str(c) for c in (
            master_df["theoretical_factors"].tolist()
            + master_df["detected_abbrev"].tolist()
            + master_df["empirical_factors"].tolist()
        ))
    ]
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              frameon=True, fontsize=11, title="Factor", title_fontsize=11,
              bbox_to_anchor=(0.5, -0.02), bbox_transform=ax.transAxes)

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Alluvial figure saved → {output_path}")


def _draw_bezier(ax, x0, y0, x1, y1, color, alpha=0.35, lw=1.0):
    """Draw a smooth cubic Bezier curve between two points."""
    cx = (x0 + x1) / 2
    verts = [(x0, y0), (cx, y0), (cx, y1), (x1, y1)]
    codes = [1, 4, 4, 4]  # MOVETO, CURVE4 * 3
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


# ── entry point ──────────────────────────────────────────────────────────────
def resolve_paths(args, root):
    run_dir = Path(args.run_dir) if args.run_dir else (
        root / "data" / "outputs"
        / "step2_b4_validation"
        / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
    )
    ref_models = Path(args.reference_models) if args.reference_models else root / "config" / "reference_models_b4.yaml"
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "outputs" / "concordance" / "b4"
    return run_dir, ref_models, output_dir


def main():
    parser = argparse.ArgumentParser(description="Item-level assignment analysis")
    parser.add_argument("--run-dir")
    parser.add_argument("--reference-models")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir, ref_path, out_dir = resolve_paths(args, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    items_df = pd.read_csv(run_dir / "items.csv")
    cluster_df = pd.read_csv(run_dir / "03_clustering" / "cluster_assignments.csv")
    cluster_label_map = load_cluster_labels(run_dir / "04_labeling" / "cluster_concept_labels.csv")
    ref_models = load_reference_models(ref_path)

    theo_dict = {k: v for k, v in ref_models["theoretical"].items() if isinstance(v, list)}
    emp_dict = {k: v for k, v in ref_models["empirical"].items() if isinstance(v, list)}

    print("Building master table…")
    master_df = build_master_df(items_df, cluster_df, cluster_label_map, theo_dict, emp_dict)

    # Load all B4 items from txt; build unmodelled subset
    b4_txt = root / "data" / "raw" / "items_description" / "b4_cog_items.txt"
    all_b4 = load_all_b4_items_from_txt(b4_txt) if b4_txt.exists() else {}
    modelled_ids = set(master_df["item_id"].tolist())
    unmodelled_rows = []
    for iid in sorted(all_b4.keys()):
        if iid not in modelled_ids:
            info = all_b4[iid]
            unmodelled_rows.append({
                "item_id": iid,
                "item_title": info["item_title"],
                "item_description": info["item_description"],
                "scoring_summary": info["scoring_summary"],
                "detected_cluster_num": "",
                "detected_cluster_label": "not modelled",
                "detected_abbrev": "",
                "detected_probability": "",
                "theoretical_factors": "",
                "empirical_factors": "",
                "theo_det_agree": "",
                "emp_det_agree": "",
                "theo_emp_overlap": "",
            })
    unmodelled_df = pd.DataFrame(unmodelled_rows) if unmodelled_rows else None

    print("Writing Excel workbook…")
    write_excel(master_df, out_dir / "item_assignments.xlsx", unmodelled_df=unmodelled_df)

    print("Rendering alluvial figure…")
    (root / "results" / "figures" / "supplementary").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    plot_alluvial(master_df, root / "results" / "figures" / "supplementary" / "figS3_b4_threeway_alluvial.png")

    # summary
    n_agree = master_df["theo_det_agree"].sum()
    n_total = len(master_df)
    print(f"\nSummary: {n_agree}/{n_total} items where Detected agrees with Theoretical")
    print(f"Outputs in: {out_dir}")

    # ── Export structured metrics ────────────────────────────────────────────
    results_dir = root / "results" / "metrics"
    ensure_dir(str(results_dir))
    metrics_file = str(results_dir / "manuscript_metrics.json")

    # Item-level agreement uses the ANY-FACTOR rule: an item counts as
    # recovered when its cluster matches any expert domain listed for it
    # (the three multi-factor items COG_035/048/051 are credited to every
    # domain they are listed under; COG_047 has no expert domain and never
    # agrees).  The manuscript reports recovery over the 39 items that carry
    # an expert domain (24/39, 61.5 %); the fraction over all 40 clustered
    # items (24/40, 60.0 %) is kept under the *_all_clustered keys.
    n_agree_int = int(master_df["theo_det_agree"].sum())
    has_theo = master_df["theoretical_factors"] != "—"
    n_expert = int(has_theo.sum())
    n_multi = int(master_df["theoretical_factors"].str.contains(",", na=False).sum())
    prim_agree = int((master_df.loc[has_theo, "theoretical_factors"].str.split(", ").str[0]
                      == master_df.loc[has_theo, "detected_abbrev"]).sum())
    domain_acc = {}
    for factor in CANONICAL_THEO_ORDER:
        sub = master_df[master_df["theoretical_factors"].str.contains(factor, na=False)]
        if len(sub) > 0:
            # recovered INTO this domain (not merely into one of its domains)
            n_f_agree = int((sub["detected_abbrev"] == factor).sum())
            single = sub[~sub["theoretical_factors"].str.contains(",", na=False)]
            domain_acc[factor] = {
                "n_items": len(sub),
                "n_agree": n_f_agree,
                "pct_agreement": round(100 * n_f_agree / len(sub), 1),
                "n_items_single_domain": len(single),
                "n_agree_single_domain": int((single["detected_abbrev"] == factor).sum()),
            }
    update_results_json("Bayley4_ItemLevel_Theo_vs_Det", {
        "overall_agreement_rate_pct": round(100 * n_agree_int / n_expert, 1),
        "overall_agreement_fraction": f"{n_agree_int}/{n_expert}",
        "overall_agreement_rate_pct_all_clustered": round(100 * n_agree_int / n_total, 1),
        "overall_agreement_fraction_all_clustered": f"{n_agree_int}/{n_total}",
        "agreement_rule": "any-factor: cluster matches any expert domain listed for the item; "
                          "denominator = the 39 items with an expert domain (COG_047 has none)",
        "n_items_clustered": int(n_total),
        "n_items_with_expert_domain": int(has_theo.sum()),
        "n_items_multi_factor": n_multi,
        "items_without_expert_domain": sorted(master_df.loc[~has_theo, "item_id"].tolist()),
        "n_discordant": int(n_total - n_agree_int),
        "discordant_items": sorted(master_df.loc[master_df["theo_det_agree"] == 0, "item_id"].tolist()),
        "primary_domain_agreement_fraction": f"{prim_agree}/{int(has_theo.sum())}",
        "domain_specific_accuracy": domain_acc,
    }, filename=metrics_file)

    disagree_df = master_df[master_df["theo_det_agree"] == 0][
        ["item_id", "item_title", "theoretical_factors", "detected_abbrev", "detected_probability"]
    ].copy().sort_values("theoretical_factors")
    disagree_path = root / "results" / "tables" / "mismatches_ledger_b4.csv"
    disagree_df.to_csv(str(disagree_path), index=False)
    print(f"  Mismatches ledger → {disagree_path}")


if __name__ == "__main__":
    main()
