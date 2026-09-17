"""
lancet_style.py — shared styling layer for the Bayley-III domain-differentiation
figure suite (The Lancet Child & Adolescent Health).

This module is the *single source of truth* for colour, typography, and layout.
Every figure script imports it, so the palette is locked in exactly one place.

Colour rule (hard constraint): the five cognitive-factor hex codes below are
identical to the companion web interface and are colourblind-safe. They must
never be altered. They are copied verbatim from the existing project scripts
(figures/umap3d_beautiful.py, figures/umap3d_viewer.py) so drift is impossible.

Public helpers
--------------
mm(x)                     millimetres -> inches (Lancet column widths)
set_lancet_rcparams()     apply the house matplotlib style
decimal_formatter()       tick formatter using the mid-line decimal "·"
fmt_num(x)                format a single number Lancet-style ("0·846", "−0·37")
panel_label(ax, "A")      bold upper-left panel letter
factor_header(ax, label, color)   coloured facet-strip header above a panel
sig_stars(p)              "***" / "**" / "*" / "†" / ""
save_figure(fig, stem, outdir)    dual/triple save (pdf + png + svg), fonts embedded
sankey(ax, matrix, ...)   native (vector) matplotlib alluvial / Sankey ribbons
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches


# ============================================================================
# 1. LOCKED PALETTE  (do not edit the five factor hexes)
# ============================================================================
FACTOR_COLORS: dict[str, str] = {
    "WM":   "#2D5FA3",   # Working Memory        — royal blue
    "FS":   "#D4783A",   # Flexibility / Shift    — warm amber
    "GDPS": "#2E9E8A",   # Goal-Directed Prob.Sol — teal
    "ATT":  "#B5404E",   # Attention              — brick rose
    "HOP":  "#7755A8",   # Higher-Order Proc.     — soft violet
}
# Auxiliary category colours (also from the web interface / viewer)
MULTI_FACTOR_COLOR = "#C2478A"    # multi-factor expert calls
UC_COLOR           = "#8a94a6"    # unclassified cluster
UNASSIGNED_COLOR   = "#aaaaaa"
# "General cognition" composite is NOT one of the five factors; give it a
# neutral slate so it reads as the pooled reference, never mistaken for a factor.
GENERAL_COG_COLOR  = "#6B7280"

# Per-factor marker glyphs (kept consistent with umap3d_beautiful.py)
MARKERS: dict[str, str] = {"WM": "o", "FS": "s", "GDPS": "^", "ATT": "D", "HOP": "P"}

FACTOR_ORDER = ["WM", "FS", "GDPS", "ATT", "HOP"]

# Alias map: the MIMIC/CFA data code the goal-directed factor as "GDP" and the
# composite as "COG"; normalise everything back to the canonical palette keys.
_FACTOR_ALIASES = {
    "GDP": "GDPS", "GDPS": "GDPS",
    "WM": "WM", "FS": "FS", "ATT": "ATT", "HOP": "HOP",
    "COG": "GENERAL", "GENERAL": "GENERAL", "UNIDIM": "GENERAL",
}


def factor_color(name: str) -> str:
    """Return the locked colour for a factor code (accepts GDP/COG aliases)."""
    key = _FACTOR_ALIASES.get(str(name).upper(), str(name).upper())
    if key == "GENERAL":
        return GENERAL_COG_COLOR
    return FACTOR_COLORS.get(key, UC_COLOR)


# Predictor-domain palette for the MIMIC figure (from plot_mimic_pub.R)
PREDICTOR_DOMAIN_COLORS: dict[str, str] = {
    "Neonatal":         "#2166ac",
    "Obstetric":        "#d73027",
    "Sociodemographic": "#1a9641",
}
# Outcome-domain palette (from plot_outcomes_pub.R)
OUTCOME_DOMAIN_COLORS: dict[str, str] = {
    "Bayley":      "#1a9641",
    "CBCL Broad":  "#B5404E",
    "CBCL Narrow": "#E08A3C",
    "Q-CHAT":      "#7755A8",
}
# Pairing-stage palette (from raincloud_pairing.py)
PAIRING_COLORS = {"title_locked": "#1B5FA8", "fulltext": "#E07B39"}


# ============================================================================
# 2. GEOMETRY  (Lancet column widths)
# ============================================================================
SINGLE_COL_MM = 89.0     # single column
ONEHALF_MM    = 120.0    # 1.5 column
DOUBLE_COL_MM = 183.0    # full width (2 columns)


def mm(x: float) -> float:
    """Millimetres to inches."""
    return x / 25.4


# ============================================================================
# 3. TYPOGRAPHY  (house matplotlib style)
# ============================================================================
def set_lancet_rcparams(base: float = 8.0) -> None:
    """Apply the Lancet house style. `base` is the body font size in points."""
    matplotlib.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "Helvetica Neue", "Liberation Sans", "DejaVu Sans"],
        "font.size":         base,
        "axes.titlesize":    base + 2,
        "axes.labelsize":    base,
        "xtick.labelsize":   base - 0.5,
        "ytick.labelsize":   base - 0.5,
        "legend.fontsize":   base - 0.5,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#1a1a1a",
        "text.color":        "#1a1a1a",
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,     # embed TrueType (editable text)
        "ps.fonttype":       42,
        "svg.fonttype":      "none", # keep text as text in SVG
        "figure.dpi":        150,
    })


# ============================================================================
# 4. NUMBER FORMATTING  (mid-line decimal, true minus)
# ============================================================================
MINUS = "−"      # − true minus sign
MIDDOT = "·"     # · mid-line decimal point


def fmt_num(x: float, decimals: int = 2, strip_zero: bool = False) -> str:
    """Format a number Lancet-style: mid-line decimal + true minus.

    strip_zero=True renders |x|<1 without the leading zero (e.g. ·15).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = f"{x:.{decimals}f}"
    if strip_zero:
        s = s.replace("0.", ".", 1) if s.startswith("0.") else s
        s = s.replace("-0.", "-.", 1) if s.startswith("-0.") else s
    s = s.replace("-", MINUS).replace(".", MIDDOT)
    return s


def decimal_formatter(decimals: int | None = None, strip_zero: bool = False) -> FuncFormatter:
    """Axis tick formatter that uses the mid-line decimal and true minus.

    If `decimals` is None, integers print without a decimal part.
    """
    def _f(v, _pos):
        if decimals is None:
            s = f"{v:g}"
        else:
            s = f"{v:.{decimals}f}"
        if strip_zero and (s.startswith("0.") or s.startswith("-0.")):
            s = s.replace("0.", ".", 1)
        return s.replace("-", MINUS).replace(".", MIDDOT)
    return FuncFormatter(_f)


# ============================================================================
# 5. PANEL FURNITURE
# ============================================================================
def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.04,
                size: float = 12.0, weight: str = "bold") -> None:
    """Bold upper-left panel letter, in axes-fraction coords."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=size, fontweight=weight, color="#111111", clip_on=False)


def factor_header(ax, label: str, color: str, height: float = 0.055,
                  text_color: str = "white", fontsize: float = 9.5) -> None:
    """Draw a coloured facet-strip header spanning the top of `ax`.

    The strip sits just above the axes; its fill uses the locked factor colour,
    reinforcing factor identity while point colour can encode another variable.
    """
    ax.add_patch(mpatches.Rectangle(
        (0.0, 1.0), 1.0, height, transform=ax.transAxes,
        facecolor=color, edgecolor="none", clip_on=False, zorder=5))
    ax.text(0.5, 1.0 + height / 2.0, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=text_color, clip_on=False, zorder=6)


def sig_stars(p: float) -> str:
    """Return significance star tag for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.010:
        return "**"
    if p < 0.050:
        return "*"
    if p < 0.100:
        return "†"     # dagger
    return ""


# Legend proxy handles for the significance convention (filled vs open dots)
def sig_legend_handles(color: str = "#444444") -> list:
    return [
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor=color,
                   markeredgecolor=color, markersize=6, label="p < ·05"),
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=color, markersize=6, label="not significant"),
    ]


# ============================================================================
# 6. SAVE
# ============================================================================
def save_figure(fig, stem: str, outdir: Path | str,
                formats: Sequence[str] = ("pdf", "png", "svg"),
                dpi: int = 400) -> None:
    """Save `fig` as `<stem>.<fmt>` for each format into `outdir`."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        print(f"  saved: {path}")


SUBMISSION_FORMATS: tuple[str, ...] = ("pdf", "svg", "png")
SUBMISSION_DPI = 300      # the journal's minimum for raster copies


def save_submission(number: int, combined_stem: str, main_dir: Path | str,
                    panels: dict, outdir: Path | str) -> list[Path]:
    """Write the submission set of one multi-part figure into `outdir`:

        figure<N>_combined.{pdf,svg,png}   byte-for-byte copies of the files in
                                           `main_dir` (<combined_stem>.<fmt>)
        figure<N>_<letter>.{pdf,svg,png}   each panel figure in `panels`
                                           ({letter: Figure}), PNG at 300 dpi

    The journal asks for "the individual parts as well as a combined version"
    of every multi-part figure (Information for Authors, Figures).
    """
    import shutil
    main_dir, outdir = Path(main_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in SUBMISSION_FORMATS:
        src = main_dir / f"{combined_stem}.{fmt}"
        dst = outdir / f"figure{number}_combined.{fmt}"
        shutil.copyfile(src, dst)
        written.append(dst)
    for letter, fig in panels.items():
        for fmt in SUBMISSION_FORMATS:
            dst = outdir / f"figure{number}_{letter}.{fmt}"
            fig.savefig(dst, format=fmt, dpi=SUBMISSION_DPI, bbox_inches="tight",
                        facecolor="white")
            written.append(dst)
    for p in written:
        print(f"  saved: {p}")
    return written


# ============================================================================
# 7. NATIVE VECTOR SANKEY  (so multi-panel figures stay one vector PDF)
# ============================================================================
def _stack(sizes: Sequence[float], total_height: float, gap_frac: float = 0.02):
    """Return (y_bottom, height) for each node, stacked top->bottom with gaps."""
    sizes = np.asarray(sizes, dtype=float)
    n = len(sizes)
    gap = gap_frac * total_height
    avail = total_height - gap * (n - 1)
    scale = avail / sizes.sum() if sizes.sum() > 0 else 0.0
    heights = sizes * scale
    tops = np.zeros(n)
    y = total_height
    for i in range(n):
        y -= heights[i]
        tops[i] = y
        y -= gap
    return tops, heights


def _ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.55):
    """Draw one filled cubic-Bezier ribbon from a left slot to a right slot."""
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0), (xm, y0), (xm, y1), (x1, y1),          # top edge L->R
        (x1, y1 - h1), (xm, y1 - h1), (xm, y0 - h0), (x0, y0 - h0),  # bottom R->L
        (x0, y0),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha, zorder=1))


def sankey(ax, matrix, row_labels: Sequence[str], col_labels: Sequence[str],
           row_colors: Sequence[str], col_colors: Sequence[str],
           node_w: float = 0.06, gap_frac: float = 0.03,
           label_fs: float = 8.5, ribbon_alpha: float = 0.5) -> None:
    """Draw a two-column weighted Sankey/alluvial in `ax` (data coords 0..1).

    matrix[i, j] = flow count from row i (left) to col j (right).
    Ribbons are coloured by their source row. Fully vector (PathPatch).
    """
    M = np.asarray(matrix, dtype=float)
    nrow, ncol = M.shape
    x_left, x_right = 0.0 + node_w, 1.0 - node_w

    row_tot = M.sum(axis=1)
    col_tot = M.sum(axis=0)
    row_top, row_h = _stack(row_tot, 1.0, gap_frac)
    col_top, col_h = _stack(col_tot, 1.0, gap_frac)

    # running offsets within each node as ribbons are consumed
    row_cursor = row_top + row_h            # start at top of each row node
    col_cursor = col_top + col_h

    # draw ribbons in a stable order (row-major)
    for i in range(nrow):
        for j in range(ncol):
            v = M[i, j]
            if v <= 0:
                continue
            frac_r = v / row_tot[i] if row_tot[i] else 0
            frac_c = v / col_tot[j] if col_tot[j] else 0
            h_r = frac_r * row_h[i]
            h_c = frac_c * col_h[j]
            row_cursor[i] -= h_r
            col_cursor[j] -= h_c
            _ribbon(ax, x_left, row_cursor[i] + h_r, h_r,
                    x_right, col_cursor[j] + h_c, h_c,
                    row_colors[i], alpha=ribbon_alpha)

    # node rectangles + labels
    for i in range(nrow):
        ax.add_patch(mpatches.Rectangle((x_left - node_w, row_top[i]), node_w,
                                        row_h[i], facecolor=row_colors[i],
                                        edgecolor="none", zorder=3))
        ax.text(x_left - node_w - 0.015, row_top[i] + row_h[i] / 2.0,
                f"{row_labels[i]} (n={int(row_tot[i])})", ha="right", va="center",
                fontsize=label_fs, color="#222222")
    for j in range(ncol):
        ax.add_patch(mpatches.Rectangle((x_right, col_top[j]), node_w, col_h[j],
                                        facecolor=col_colors[j], edgecolor="none",
                                        zorder=3))
        ax.text(x_right + node_w + 0.015, col_top[j] + col_h[j] / 2.0,
                f"{col_labels[j]} (n={int(col_tot[j])})", ha="left", va="center",
                fontsize=label_fs, color="#222222")

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")


if __name__ == "__main__":
    # tiny smoke test of the palette + sankey when run directly
    set_lancet_rcparams()
    print("Factor palette:", FACTOR_COLORS)
    fig, ax = plt.subplots(figsize=(mm(120), mm(70)))
    M = np.array([[8, 6, 4, 0, 0],
                  [0, 1, 20, 0, 0],
                  [0, 16, 4, 0, 6],
                  [2, 0, 0, 18, 0],
                  [5, 0, 1, 0, 0]])
    cols = [FACTOR_COLORS[k] for k in FACTOR_ORDER]
    sankey(ax, M, FACTOR_ORDER, FACTOR_ORDER, cols, cols)
    save_figure(fig, "_smoketest_sankey", Path(__file__).parent / "outputs",
                formats=("png",))
    print("ok")
