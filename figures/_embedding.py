"""
_embedding.py — panel drawers for the embedding-geometry figure (Fig 2).

Three reusable drawers, each rendering into a caller-supplied Axes so they can be
composed into a multi-panel figure or used standalone:

  draw_raincloud(ax, root)        pairing cosine-similarity raincloud
  draw_umap3d(ax3d, ...)          3-D UMAP hero with ellipsoidal cluster hulls
  draw_within_between(ax, root)   within- vs between-domain cosine lollipop

All colour comes from lancet_style (the locked palette); the UMAP geometry reuses
the helpers in _umap_helpers.py (extracted from the original exploratory
umap3d_beautiful.py script) so it is identical to the project figure.

The "translated structure" (item -> domain map used for panels B/C) is read from
results/tables/bayley3_item_domain_assignments.csv, the single source of truth
built by analysis/item_assignments.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

import lancet_style as ls

_FIGURES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FIGURES_DIR))
import _umap_helpers as ub   # noqa: E402  (wireframe/ellipsoid helpers, params)

# Committed inputs that the figure scripts read (relative to the repository root)
UMAP_CACHE = "data/outputs/figure_inputs/umap3d_coords.csv"
ASSIGNMENTS = "results/tables/bayley3_item_domain_assignments.csv"
STEP1_DIR = "data/outputs/step1_translation"
STEP3_RUN = "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"


# ============================================================================
# Panel A — pairing raincloud
# ============================================================================
def draw_raincloud(ax, root: Path):
    """Half-violin KDE + boxplot + jittered dots of B4→B3 full-text cosine sim.
    Dots coloured by pairing stage (title-locked vs full-text fallback)."""
    csv = root / "data" / "outputs" / "step1_translation" / "b4_b3_cascade_pairs.csv"
    df = pd.read_csv(csv).dropna(subset=["sim_fulltext", "pairing_stage"])
    sim = df["sim_fulltext"].to_numpy()
    stage = df["pairing_stage"].to_numpy()
    m_tl, m_ft = stage == "title_locked", stage == "fulltext"
    n = len(sim)

    mean_v, med_v = float(np.mean(sim)), float(np.median(sim))
    q1, q3 = np.percentile(sim, [25, 75])
    iqr = q3 - q1
    wlo = max(sim.min(), q1 - 1.5 * iqr)
    whi = min(sim.max(), q3 + 1.5 * iqr)

    c_kde = ls.PAIRING_COLORS["title_locked"]
    c_ft = ls.PAIRING_COLORS["fulltext"]

    KBASE, KH, BOX_Y, BOX_H, DOT_Y, JIT = 0.40, 0.52, 0.22, 0.055, 0.09, 0.036

    kde = gaussian_kde(sim, bw_method=0.18)
    xg = np.linspace(0.47, 1.02, 600)
    dens = kde(xg); dens = dens / dens.max()
    ax.fill_between(xg, KBASE, KBASE + dens * KH, color=c_kde, alpha=0.12, lw=0)
    ax.plot(xg, KBASE + dens * KH, color=c_kde, lw=0.9, solid_capstyle="round")

    # boxplot
    ax.add_patch(mpatches.Rectangle((q1, BOX_Y - BOX_H), q3 - q1, 2 * BOX_H,
                 edgecolor=c_kde, facecolor=c_kde, alpha=0.12, lw=0.8, zorder=3))
    for yy in (BOX_Y - BOX_H, BOX_Y + BOX_H):
        ax.plot([q1, q3], [yy, yy], color=c_kde, lw=0.8, zorder=4)
    for xx in (q1, q3):
        ax.plot([xx, xx], [BOX_Y - BOX_H, BOX_Y + BOX_H], color=c_kde, lw=0.8, zorder=4)
    ax.plot([wlo, q1], [BOX_Y, BOX_Y], color=c_kde, lw=0.8, zorder=4)
    ax.plot([q3, whi], [BOX_Y, BOX_Y], color=c_kde, lw=0.8, zorder=4)
    for w in (wlo, whi):
        ax.plot([w, w], [BOX_Y - BOX_H * 0.45, BOX_Y + BOX_H * 0.45],
                color=c_kde, lw=0.8, zorder=4)

    rng = np.random.default_rng(42)
    jit = rng.uniform(-JIT, JIT, n)
    ax.scatter(sim[m_tl], DOT_Y + jit[m_tl], s=9, color=c_kde, alpha=0.8,
               linewidths=0.3, edgecolors="white", zorder=5,
               label=f"Title-locked (n={int(m_tl.sum())})")
    ax.scatter(sim[m_ft], DOT_Y + jit[m_ft], s=9, color=c_ft, alpha=0.85,
               marker="D", linewidths=0.3, edgecolors="white", zorder=5,
               label=f"Full-text (n={int(m_ft.sum())})")

    ax.axvline(mean_v, color="#C0392B", lw=0.9, ls=(0, (5, 4)), ymin=0.02, ymax=0.97, zorder=6)
    ax.axvline(med_v, color="#1E8449", lw=0.9, ls=(0, (2, 3)), ymin=0.02, ymax=0.97, zorder=6)

    ax.set_xlim(0.47, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.xaxis.set_major_formatter(ls.decimal_formatter(decimals=1))
    ax.set_xlabel("Full-text cosine similarity  (Bayley-4 → Bayley-III pairs)")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        mlines.Line2D([], [], color=c_kde, marker="o", ls="none", ms=4, alpha=0.8,
                      label=f"Title-locked (n={int(m_tl.sum())})"),
        mlines.Line2D([], [], color=c_ft, marker="D", ls="none", ms=4, alpha=0.85,
                      label=f"Full-text (n={int(m_ft.sum())})"),
        mlines.Line2D([], [], color="#C0392B", lw=0.9, ls=(0, (5, 4)),
                      label=f"Mean = {ls.fmt_num(mean_v, 3)}"),
        mlines.Line2D([], [], color="#1E8449", lw=0.9, ls=(0, (2, 3)),
                      label=f"Median = {ls.fmt_num(med_v, 3)}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2, frameon=False,
              handlelength=1.8, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
    return mean_v, med_v


# ============================================================================
# Panel B — 3-D UMAP hero (into a provided 3-D axes)
# ============================================================================
def load_umap_cache(root: Path) -> pd.DataFrame:
    cache = root / UMAP_CACHE
    if not cache.exists():
        raise SystemExit("Missing UMAP cache. Run:  python figures/_umap_coords.py")
    return pd.read_csv(cache)


def draw_umap3d(ax, df: pd.DataFrame, color_col: str = "detected",
                elev: float = 55.0, azim: float = 113.0,
                hull_std: float = 1.8, dot_size: int = 40, legend: bool = True,
                hulls: bool = True, contour_ids=None):
    """Render the 3-D UMAP into 3-D axes `ax` using cached coords.
    Pass hulls=False for a points-only cloud (honest for overlapping domains)."""
    coords = df[["x", "y", "z"]].to_numpy()
    ax.set_facecolor("white")
    ax.set_axis_off()

    pad = 0.18
    xl = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    yl = (coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    zl = (coords[:, 2].min() - pad, coords[:, 2].max() + pad)
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)

    ub._draw_wireframe_box(ax, xl, yl, zl)

    for f in (ub.FACTOR_ORDER if hulls else []):
        pts = coords[df[color_col].to_numpy() == f]
        if len(pts) >= 4:
            ub._draw_ellipsoid(ax, pts.mean(axis=0), np.cov(pts.T),
                               ls.FACTOR_COLORS[f], n_std=hull_std, alpha=0.14)

    cset = set(contour_ids) if contour_ids is not None else set()
    ids = df["item_id"].to_numpy() if "item_id" in df.columns else np.array([None] * len(df))
    for f in ub.FACTOR_ORDER:
        m = df[color_col].to_numpy() == f
        if not m.any():
            continue
        pts = coords[m]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=ls.FACTOR_COLORS[f],
                   marker=ub.MARKERS[f], s=dot_size, alpha=0.92,
                   edgecolors="white", linewidths=0.6, depthshade=True, zorder=4)
        # extended/uncertain (UC) items: soft grey halo ring so they can be eyeballed
        mc = m & np.isin(ids, list(cset))
        if mc.any():
            cp = coords[mc]
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], facecolors="none",
                       edgecolors="#9a9a9a", linewidths=1.0, s=dot_size * 2.4,
                       marker="o", depthshade=False, zorder=3)

    if legend:
        handles = [mpatches.Patch(facecolor=ls.FACTOR_COLORS[f], edgecolor="white",
                                  linewidth=0.5, label=f)
                   for f in ub.FACTOR_ORDER if (df[color_col].to_numpy() == f).any()]
        if cset:
            handles.append(mlines.Line2D([], [], marker="o", color="none",
                           markerfacecolor="none", markeredgecolor="#9a9a9a",
                           markeredgewidth=1.0, markersize=8, label="Extended"))
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False,
                  labelspacing=0.35, handlelength=1.0, handletextpad=0.4,
                  bbox_to_anchor=(-0.02, 1.0))
    ax.view_init(elev=elev, azim=azim)


def draw_umap2d(ax, df: pd.DataFrame, color_col: str = "detected",
                n_std: float = 1.8, dot_size: int = 46, legend: bool = False,
                hulls: bool = True, contour_ids=None):
    """2-D UMAP with covariance-ellipse cluster hulls (print-safe alternative to 3-D).
    Pass hulls=False for a points-only scatter (honest for overlapping domains).

    Requires the x2/y2 columns from _umap_coords.py (re-run it if missing)."""
    if "x2" not in df.columns:
        raise SystemExit("UMAP cache lacks 2-D coords. Re-run: python figures/_umap_coords.py")
    coords = df[["x2", "y2"]].to_numpy()

    for f in (ub.FACTOR_ORDER if hulls else []):
        pts = coords[df[color_col].to_numpy() == f]
        if len(pts) < 3:
            continue
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        vals = np.maximum(vals, 0.18 * vals.max())   # avoid needle hulls (e.g. HOP, n=6)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle,
                     facecolor=ls.FACTOR_COLORS[f], edgecolor=ls.FACTOR_COLORS[f],
                     alpha=0.12, lw=0.8, zorder=1))

    cset = set(contour_ids) if contour_ids is not None else set()
    ids = df["item_id"].to_numpy() if "item_id" in df.columns else np.array([None] * len(df))
    for f in ub.FACTOR_ORDER:
        m = df[color_col].to_numpy() == f
        if not m.any():
            continue
        pts = coords[m]
        ax.scatter(pts[:, 0], pts[:, 1], c=ls.FACTOR_COLORS[f], marker=ub.MARKERS[f],
                   s=dot_size, alpha=0.92, edgecolors="white", linewidths=0.6, zorder=3,
                   label=f)
        # extended/uncertain (UC) items: soft grey halo ring so they can be eyeballed
        mc = m & np.isin(ids, list(cset))
        if mc.any():
            cp = coords[mc]
            ax.scatter(cp[:, 0], cp[:, 1], facecolors="none", edgecolors="#9a9a9a",
                       linewidths=1.0, s=dot_size * 2.3, marker="o", zorder=2)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
    ax.set_ylabel("UMAP 2", fontsize=8, labelpad=2)
    for s in ax.spines.values():
        s.set_color("#cfcfcf"); s.set_linewidth(0.8)
    ax.margins(0.08)
    if legend:
        handles = [mlines.Line2D([], [], marker=ub.MARKERS[f], color="none",
                    markerfacecolor=ls.FACTOR_COLORS[f], markeredgecolor="white",
                    markeredgewidth=0.5, markersize=5.5, label=f)
                   for f in ub.FACTOR_ORDER if (df[color_col].to_numpy() == f).any()]
        if cset:
            handles.append(mlines.Line2D([], [], marker="o", color="none",
                           markerfacecolor="none", markeredgecolor="#9a9a9a",
                           markeredgewidth=1.0, markersize=8, label="Extended"))
        ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False,
                  labelspacing=0.3, handlelength=1.0, handletextpad=0.4, borderaxespad=0.4)


# ============================================================================
# Panel C — within- vs between-domain cosine lollipop
# ============================================================================
_FACTOR_SHORT = {
    "attention": "ATT", "working_memory": "WM",
    "goal_directed_problem_solving": "GDPS", "flexibility_shift": "FS",
    "higher_order_processing": "HOP",
}
_ORDER = ["ATT", "WM", "GDPS", "FS", "HOP"]


def uc_item_ids(root: Path) -> set:
    """Item ids flagged as uncertain (UC) by the leave-one-out centroid procedure
    (assigned_factor == uncategorized_cognition in the mixed_centroids solution)."""
    up = pd.read_csv(root / STEP1_DIR / "mixed_centroids/all_unpaired_b3_assignments.csv")
    return set(up.loc[up["assigned_factor"] == "uncategorized_cognition", "item_id"])


def extended_item_ids(root: Path) -> set:
    """All items added to the model AFTER the direct cascade pairing — i.e. every
    unpaired item allocated by the centroid procedure, both the ones with a clear
    factor preference and the UC-flagged uncertain ones (the 52 unpaired items)."""
    a = pd.read_csv(root / ASSIGNMENTS)
    return set(a.loc[a["assignment_source"] != "cascade_pairing", "item_id"])


def translated_short_map(root: Path) -> dict:
    """item_id → short factor (ATT/WM/GDPS/FS/HOP) for the *translated* model
    (cascade pairing + centroid assignment + expert review of the UC-flagged
    items), read from the committed taxonomy table.  Distribution:
    ATT 22 / WM 19 / GDPS 23 / FS 20 / HOP 7."""
    path = root / ASSIGNMENTS
    if not path.exists():
        raise SystemExit(f"Missing {ASSIGNMENTS}. Run:  python analysis/item_assignments.py")
    a = pd.read_csv(path)
    return dict(zip(a["item_id"], a["domain"]))


def detected_short_map(root: Path) -> dict:
    """item_id → short factor for the bottom-up (Objective 3) solution: consensus
    cluster of each item, labelled post hoc by the nearest domain name
    (04_labeling/cluster_concept_labels.csv)."""
    run = root / STEP3_RUN
    clusters = pd.read_csv(run / "03_clustering/cluster_assignments.csv")[["item_id", "cluster"]]
    labels = pd.read_csv(run / "04_labeling/cluster_concept_labels.csv")
    norm = {"working memory": "WM", "attention": "ATT", "higher order processes": "HOP",
            "higher order processing": "HOP", "flexibility shift": "FS",
            "goal directed problem solving": "GDPS"}
    cid2short = {int(r.cluster_id): norm[str(r.top_concept).strip().lower()] for r in labels.itertuples()}
    return {r.item_id: cid2short[int(r.cluster)] for r in clusters.itertuples()}


def _assignments(root: Path, solution: str):
    emb_dir = root / STEP3_RUN
    items = pd.read_csv(emb_dir / "items.csv")
    emb = np.load(emb_dir / "embeddings_all_mpnet_base_v2.npy")
    emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    sim = emb_n @ emb_n.T

    tmap = translated_short_map(root) if solution == "theoretical" else detected_short_map(root)
    fs = items["item_id"].map(tmap).values
    return sim, fs


def panel_c_table(root: Path, solution: str = "theoretical") -> pd.DataFrame:
    """Return the numeric values plotted in panel C as a tidy table.

    Rows contain within-domain means and all directed between-domain means for each
    present factor.
    """
    sim, fs = _assignments(root, solution)
    present = [f for f in _ORDER if (fs == f).any()]
    counts = {f: int((fs == f).sum()) for f in present}

    def within(fa):
        idx = np.where(fs == fa)[0]
        if len(idx) < 2:
            return np.nan
        block = sim[np.ix_(idx, idx)]
        return float(block[np.triu_indices(len(idx), 1)].mean())

    def between(fa, fb):
        ia, ib = np.where(fs == fa)[0], np.where(fs == fb)[0]
        if len(ia) == 0 or len(ib) == 0:
            return np.nan
        return float(sim[np.ix_(ia, ib)].mean())

    rows = []
    for foc in present:
        ni = counts[foc]
        rows.append({
            "solution": solution,
            "focus_factor": foc,
            "comparison": "within",
            "other_factor": np.nan,
            "n_focus": ni,
            "n_other": np.nan,
            "pair_count": int(ni * (ni - 1) / 2),
            "mean_cosine": within(foc),
        })
        for other in present:
            if other == foc:
                continue
            nj = counts[other]
            rows.append({
                "solution": solution,
                "focus_factor": foc,
                "comparison": "between",
                "other_factor": other,
                "n_focus": ni,
                "n_other": nj,
                "pair_count": int(ni * nj),
                "mean_cosine": between(foc, other),
            })

    cols = [
        "solution",
        "focus_factor",
        "comparison",
        "other_factor",
        "n_focus",
        "n_other",
        "pair_count",
        "mean_cosine",
    ]
    return pd.DataFrame(rows, columns=cols)


def draw_within_between(ax, root: Path, solution: str = "theoretical", title: str | None = None):
    """Cleveland lollipop: within-domain mean cosine (large filled) vs between-domain
    means to each other domain (small, coloured by the other domain)."""
    sim, fs = _assignments(root, solution)

    def within(fa):
        idx = np.where(fs == fa)[0]
        if len(idx) < 2:
            return np.nan
        block = sim[np.ix_(idx, idx)]
        return float(block[np.triu_indices(len(idx), 1)].mean())

    def between(fa, fb):
        ia, ib = np.where(fs == fa)[0], np.where(fs == fb)[0]
        if len(ia) == 0 or len(ib) == 0:
            return np.nan
        return float(sim[np.ix_(ia, ib)].mean())

    present = [f for f in _ORDER if (fs == f).any()]
    nf = len(present)
    ymap = {f: nf - 1 - i for i, f in enumerate(present)}

    for foc in present:
        yi = ymap[foc]
        col = ls.FACTOR_COLORS[foc]
        ni = int((fs == foc).sum())
        wm = within(foc)
        bvs = {o: between(foc, o) for o in present if o != foc}
        valid = [v for v in bvs.values() if not np.isnan(v)]
        ax.axhspan(yi - 0.45, yi + 0.45, color="0.97", lw=0, zorder=0)
        if valid and not np.isnan(wm):
            ax.hlines(yi, min(valid), wm, color="0.75", lw=1.1, zorder=1)
        for other, bv in bvs.items():
            if not np.isnan(bv):
                ax.scatter(bv, yi, s=48, color=ls.FACTOR_COLORS[other],
                           edgecolors="white", linewidths=0.7, alpha=0.9, zorder=3)
        if not np.isnan(wm):
            ax.scatter(wm, yi, s=120, color=col, edgecolors="black", linewidths=1.0, zorder=5)
        ax.text(-0.015, yi, f"{foc} (n={ni})", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, fontweight="bold", color=col,
                clip_on=False)

    ax.set_yticks([])
    ax.set_xlabel("Mean pairwise cosine similarity")
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(ls.decimal_formatter(decimals=2))
    ax.set_ylim(-0.6, nf - 0.4)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [
        mlines.Line2D([], [], marker="o", color="0.4", markeredgecolor="black",
                      markeredgewidth=1.0, markersize=9, ls="none", label="Within-domain (mean)"),
        mlines.Line2D([], [], marker="o", color="0.4", markeredgecolor="white",
                      markeredgewidth=0.7, markersize=6, ls="none", label="Between-domain"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False,
              handlelength=1.0, handletextpad=0.5, labelspacing=0.3)
