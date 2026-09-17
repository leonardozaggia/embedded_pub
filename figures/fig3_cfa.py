"""
Figure 3 — The validated three-factor measurement model.

The clinical results rest on a 23-item CFA (WM, GDPS, FS) that fits the dHCP data
substantially better than a unidimensional composite — a claim the paper states
but never shows. This figure fills that gap:

  A  Standardised CFA loadings (STDYX) for all 23 items, grouped and coloured by
     factor, with ±1 SE whiskers.
  B  Miniature measurement-model path diagram: three latent factors (ellipses) over
     their observed items (squares), with loading lines and inter-factor correlation
     arcs (WM–GDPS, GDPS–FS, WM–FS).
  C  Distribution of EAP factor scores per domain (WM, GDPS, FS).
  D  Global fit: three-factor vs unidimensional model across RMSEA, SRMR, CFI, TLI.

Loadings and correlations are parsed from the Mplus STDYX output of the
three-factor model (model_cfa_3factor.out); the unidimensional fit indices
(panel D's open markers) are parsed from model_unidimensional.out.

Usage:
    python figures/fig3_cfa.py            # results/figures/main/fig3_cfa.*
    python figures/fig3_cfa.py --panels   # + results/figures/submission/figure3_{A,B,C,D,combined}.*
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
import lancet_style as ls

CFA_OUT = ROOT / "psychometrics" / "measurement" / "model_cfa_3factor.out"
UNIDIM_OUT = ROOT / "psychometrics" / "measurement" / "model_unidimensional.out"
FSCORES = ROOT / "psychometrics" / "fscores_measurement.dat"
MAIN_DIR = ROOT / "results" / "figures" / "main"
SUBMISSION_DIR = ROOT / "results" / "figures" / "submission"
STEM = "fig3_cfa"
FAC_DISPLAY = {"WM": "WM", "GDP": "GDPS", "FS": "FS"}     # Mplus GDP → display GDPS
FAC_ORDER = ["WM", "GDP", "FS"]

# EAP factor-score columns in fscores_measurement.dat.
FSCORE_COL = {"WM": 23, "GDPS": 25, "FS": 27}


# ── Parse global fit indices from any Mplus .out ─────────────────────────────
def parse_fit(path: Path):
    text = path.read_text(errors="ignore").splitlines()

    def _grab(anchor, key):
        for i, l in enumerate(text):
            if anchor in l:
                for l2 in text[i:i + 6]:
                    mm = re.search(rf"{key}\s+([0-9.]+)", l2)
                    if mm:
                        return float(mm.group(1))
        return None
    return {
        "rmsea": _grab("RMSEA (Root", "Estimate"),
        "cfi":   _grab("CFI/TLI", "CFI"),
        "tli":   _grab("CFI/TLI", "TLI"),
        "srmr":  _grab("SRMR (Standardized", "Value"),
    }


# ── Parse the Mplus STDYX section ────────────────────────────────────────────
def parse_cfa(path: Path):
    text = path.read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(text) if "STDYX Standardization" in l)
    loadings = {f: [] for f in FAC_ORDER}
    corrs = {}
    mode = cur = None
    for line in text[start + 1:]:
        st = line.strip()
        # stop at the end of the StdYX block
        if "Standardization" in line or st in ("R-SQUARE", "QUALITY OF NUMERICAL RESULTS"):
            break
        # section headers end the BY/WITH context
        if re.match(r"^(Means|Intercepts|Thresholds|Variances|Residual Variances|Scales)\b", st):
            mode = None
            continue
        m = re.match(r"^([A-Z]{2,4})\s+(BY|WITH)\s*$", st)
        if m:
            cur, mode = m.group(1), m.group(2)
            continue
        p = st.split()
        if mode == "BY" and p and p[0].startswith("COG") and len(p) >= 3:
            loadings[cur].append((p[0], float(p[1]), float(p[2])))
        elif mode == "WITH" and p and p[0] in FAC_ORDER and p[0] != cur and len(p) >= 2:
            corrs.setdefault((cur, p[0]), float(p[1]))

    return loadings, corrs, parse_fit(path)


# ── Panel A — loadings ───────────────────────────────────────────────────────
def panel_loadings(ax, loadings):
    rows = []                       
    for f in FAC_ORDER:
        for item, est, se in loadings[f]:
            rows.append((item, est, se, f))
    n = len(rows)
    ys = list(range(n - 1, -1, -1))
    
    counts = [len(loadings[f]) for f in FAC_ORDER]
    bounds = np.cumsum(counts)[:-1]
    for b in bounds:
        ax.axhline(ys[b] + 0.5, color="#dcdcdc", lw=0.6, zorder=0)

    ax.axvline(0.40, color="#c9c9c9", lw=0.6, ls=(0, (4, 3)), zorder=0)
    for (item, est, se, f), y in zip(rows, ys):
        c = ls.FACTOR_COLORS[FAC_DISPLAY[f]]
        ax.plot([0, est], [y, y], color=c, lw=1.6, alpha=0.85, solid_capstyle="round", zorder=2)
        ax.plot([est - se, est + se], [y, y], color=c, lw=0.9, alpha=0.7, zorder=2)
        ax.scatter([est], [y], s=30, facecolor=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(-0.02, y, "COG " + item[3:].lstrip("0"), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.6, color="#333", clip_on=False)

    idx = 0
    for f, cnt in zip(FAC_ORDER, counts):
        yc = float(np.mean(ys[idx:idx + cnt]))
        idx += cnt
        ax.text(1.008, yc, FAC_DISPLAY[f], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=ls.FACTOR_COLORS[FAC_DISPLAY[f]], clip_on=False, rotation=90)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(ls.decimal_formatter(decimals=2))
    ax.set_xlabel("Standardised loading", fontsize=8.5)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)


# ── Panel B — miniature CFA path diagram ─────────────────────────────────────
def panel_path(ax, loadings, corrs):
    def rval(a, b):
        for (x, y), v in corrs.items():
            if {FAC_DISPLAY[x], FAC_DISPLAY[y]} == {a, b}:
                return v
        return np.nan

    facs = FAC_ORDER                     
    gap = 1.7                            
    slot = 0.0
    item_slots, fac_center = {}, {}
    for f in facs:
        slots = []
        for _ in loadings[f]:
            slots.append(slot); slot += 1.0
        item_slots[f] = slots
        fac_center[f] = (slots[0] + slots[-1]) / 2.0
        slot += gap
    max_slot = slot - gap - 1.0
    x0, x1 = 0.07, 0.93
    def X(s):
        return x0 + (s / max_slot) * (x1 - x0) if max_slot > 0 else 0.5

    y_fac, y_item = 0.54, 0.08
    ew, eh = 0.155, 0.15                 

    for f in facs:
        c = ls.FACTOR_COLORS[FAC_DISPLAY[f]]
        fx = X(fac_center[f])
        for (item, est, se), s in zip(loadings[f], item_slots[f]):
            ax.plot([fx, X(s)], [y_fac - eh / 2, y_item + 0.018], color=c,
                    lw=0.3 + 1.25 * est, alpha=0.45, solid_capstyle="round", zorder=1)

    for f in facs:
        c = ls.FACTOR_COLORS[FAC_DISPLAY[f]]
        xs = [X(s) for s in item_slots[f]]
        ax.scatter(xs, [y_item] * len(xs), marker="s", s=30, facecolor=c,
                   edgecolor="white", linewidth=0.5, zorder=3)

    for f in facs:
        c = ls.FACTOR_COLORS[FAC_DISPLAY[f]]
        ax.add_patch(mpatches.Ellipse((X(fac_center[f]), y_fac), ew, eh,
                     facecolor=c, edgecolor="white", linewidth=1.1, zorder=4))
        ax.text(X(fac_center[f]), y_fac, FAC_DISPLAY[f], ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=5)

    fx = {FAC_DISPLAY[f]: X(fac_center[f]) for f in facs}
    y0 = y_fac + eh / 2
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)     # fix data range so the transforms below are valid

    # arc3 curvature is applied in DISPLAY space, so a curve's true apex height
    # depends on the axes pixel aspect (panel B is far wider than tall). Compute
    # the real apex through the data⇄display transform rather than assume square.
    inv = ax.transData.inverted()
    def true_apex_y(a, b, rad):
        pA = ax.transData.transform((fx[a], y0))
        pB = ax.transData.transform((fx[b], y0))
        mid = 0.5 * (pA + pB)
        d = pB - pA
        ctrl = mid + rad * np.array([d[1], -d[0]])          # arc3 control point (display px)
        apex = 0.25 * pA + 0.5 * ctrl + 0.25 * pB           # quadratic Bézier at t=0.5
        return inv.transform(apex)[1]

    # inner arcs sit low; the outer WM–FS arc nests clearly above them
    arcs = [("WM", "GDPS", -0.38), ("GDPS", "FS", -0.38), ("WM", "FS", -0.46)]
    for a, b, rad in arcs:
        rv = rval(a, b)
        ax.add_patch(mpatches.FancyArrowPatch(
            (fx[a], y0), (fx[b], y0),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="<->",
            mutation_scale=6, lw=0.6 + 1.9 * rv, color="#8a9099",
            alpha=0.8, zorder=2))
        if (a, b) == ("WM", "FS"):          # middle label: snug above its arc's crown
            ty, va = true_apex_y(a, b, rad) + 0.022, "bottom"
        else:                                # left / right labels: in the valley below the arcs
            ty, va = y0 + 0.004, "center"
        ax.text((fx[a] + fx[b]) / 2, ty, f"r = {ls.fmt_num(rv, 2)}",
                ha="center", va=va, fontsize=6.5, fontweight="bold",
                color="#555", zorder=6,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.axis("off")


# ── Panel C — per-facet EAP factor-score distributions ───────────────────────
def panel_scores(ax):
    data = np.loadtxt(FSCORES)
    facs = ["WM", "GDPS", "FS"]
    arrs = [data[:, FSCORE_COL[f]] for f in facs]
    arrs = [a[np.isfinite(a) & (a > -900)] for a in arrs]
    ys = [2, 1, 0]                                   
    parts = ax.violinplot(arrs, positions=ys, vert=False, widths=0.85,
                          showextrema=False, showmedians=False)
    for body, f in zip(parts["bodies"], facs):
        body.set_facecolor(ls.FACTOR_COLORS[f]); body.set_alpha(0.45)
        body.set_edgecolor(ls.FACTOR_COLORS[f]); body.set_linewidth(0.8)
    for a, y, f in zip(arrs, ys, facs):
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        ax.plot([q1, q3], [y, y], color=ls.FACTOR_COLORS[f], lw=2.4, solid_capstyle="round", zorder=4)
        ax.scatter([med], [y], s=26, facecolor="white", edgecolor=ls.FACTOR_COLORS[f],
                   linewidth=1.1, zorder=5)
        ax.text(-0.02, y, f, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, fontweight="bold", color=ls.FACTOR_COLORS[f], clip_on=False)
    ax.axvline(0, color="#c9c9c9", lw=0.6, zorder=0)
    ax.set_yticks([]); ax.set_ylim(-0.6, 2.6)
    ax.xaxis.set_major_formatter(ls.decimal_formatter(decimals=1))
    ax.set_xlabel("Factor score (θ)", fontsize=8.5)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)


# ── Panel D — fit comparison ─────────────────────────────────────────────────
def panel_fit(ax, fit3, fit1):
    indices = [
        ("RMSEA", fit3["rmsea"], fit1["rmsea"], "lower"),
        ("SRMR",  fit3["srmr"],  fit1["srmr"],  "lower"),
        ("CFI",   fit3["cfi"],   fit1["cfi"],   "higher"),
        ("TLI",   fit3["tli"],   fit1["tli"],   "higher"),
    ]
    ys = list(range(len(indices) - 1, -1, -1))
    teal = ls.FACTOR_COLORS["GDPS"]
    off = 0.040   
    dy = 0.20     
    for (name, v3, v1, better), y in zip(indices, ys):
        y3, y1 = y + dy, y - dy
        ax.plot([v1, v3], [y1, y3], color="#c4c4c4", lw=1.2, zorder=1)
        ax.scatter([v3], [y3], s=48, facecolor=teal, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.scatter([v1], [y1], s=44, facecolor="none", edgecolor="#8a8a8a",
                   linewidth=1.3, zorder=3)
        
        rightside = v3 < 0.5
        dx = off if rightside else -off
        ha = "left" if rightside else "right"
        ax.text(v3 + dx, y3, ls.fmt_num(v3, 3), ha=ha, va="center",
                fontsize=7, color=teal, fontweight="bold")
        ax.text(v1 + dx, y1, ls.fmt_num(v1, 3), ha=ha, va="center",
                fontsize=7, color="#8a8a8a")
        ax.text(-0.02, y, f"{name}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, fontweight="bold", color="#222",
                clip_on=False)
        ax.text(1.03, y, "↓ better" if better == "lower" else "↑ better",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.4, color="#999", clip_on=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(indices) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(ls.decimal_formatter(decimals=2))
    ax.set_xlabel("Fit-index value", fontsize=8.5)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        mlines.Line2D([], [], marker="o", ls="none", markerfacecolor=teal,
                      markeredgecolor="white", markersize=7, label="Three-factor"),
        mlines.Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                      markeredgecolor="#8a8a8a", markersize=7, label="Unidimensional"),
    ], loc="center", bbox_to_anchor=(0.5, 0.46), fontsize=7, frameon=False,
       handletextpad=0.4, labelspacing=0.3)


def _load():
    loadings, corrs, fit3 = parse_cfa(CFA_OUT)
    fit1 = parse_fit(UNIDIM_OUT)
    return loadings, corrs, fit3, fit1


def build(data=None):
    ls.set_lancet_rcparams(base=8.0)
    loadings, corrs, fit3, fit1 = data or _load()

    fig = plt.figure(figsize=(ls.mm(183), ls.mm(168)))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0], height_ratios=[1.05, 0.95, 1.05],
                          hspace=0.42, wspace=0.28,
                          left=0.11, right=0.93, top=0.955, bottom=0.075)
    
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])       # factor correlations
    axC = fig.add_subplot(gs[1, 1])       # factor-score distributions
    axD = fig.add_subplot(gs[2, 1])       # model fit

    panel_loadings(axA, loadings)
    panel_path(axB, loadings, corrs)
    panel_scores(axC)
    panel_fit(axD, fit3, fit1)

    for ax, letter, dx, dy in [(axA, "A", 0.060, 0.010), (axB, "B", 0.03, 0.010),
                               (axC, "C", 0.03, 0.028), (axD, "D", 0.03, 0.028)]:
        pos = ax.get_position()
        fig.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=12, fontweight="bold", va="baseline")
    return fig


def build_panels(data=None) -> dict:
    """Panels A-D as separate figures, drawn by the same panel functions."""
    ls.set_lancet_rcparams(base=8.0)
    loadings, corrs, fit3, fit1 = data or _load()
    panels = {}
    # (letter, width mm, height mm, axes rect, drawing call)
    spec = [
        ("A", 95, 168, [0.20, 0.065, 0.72, 0.92], lambda ax: panel_loadings(ax, loadings)),
        ("B", 92, 58, [0.04, 0.04, 0.92, 0.92], lambda ax: panel_path(ax, loadings, corrs)),
        ("C", 92, 52, [0.16, 0.20, 0.80, 0.76], lambda ax: panel_scores(ax)),
        ("D", 92, 58, [0.20, 0.19, 0.66, 0.77], lambda ax: panel_fit(ax, fit3, fit1)),
    ]
    for letter, w, h, rect, draw in spec:
        fig = plt.figure(figsize=(ls.mm(w), ls.mm(h)))
        draw(fig.add_axes(rect))
        panels[letter] = fig
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", action="store_true",
                    help="also write the individual panels and a copy of the combined "
                         "figure to results/figures/submission/ (figure3_A..D, "
                         "figure3_combined)")
    args = ap.parse_args()
    data = _load()
    fig = build(data)
    ls.save_figure(fig, STEM, MAIN_DIR)
    if args.panels:
        ls.save_submission(3, STEM, MAIN_DIR, build_panels(data), SUBMISSION_DIR)
    print("Figure 3 (CFA measurement model) done.")


if __name__ == "__main__":
    main()