# --- UTF-8 stdout/stderr, matching the rest of the figure suite ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

"""
fig_threshold_sensitivity.py
==============================
Supplementary Figure S8 -- justification of the 0·75 title-similarity threshold
used by the two-stage (cascade) Bayley-4 to Bayley-III item pairing.

  A  Distribution of the highest title similarity available to each of the 39
     Bayley-4 cognitive items, with the 0·75 acceptance threshold marked.
  B  Sensitivity of the pairing solution to that threshold: the number of item
     pairs that differ from the adopted 0·75 solution as the threshold is swept
     from 0·60 to 0·90 in steps of 0·01.

Both panels are computed from the real embeddings; nothing is smoothed,
simulated or idealised. The similarity matrices are rebuilt by calling the
pipeline's own functions (`parse_items`, `embed_items`, `_cascade_pairing` from
bayley_nlp.pipelines.step1_translation), so the figure cannot drift from the
analysis: on build, the recomputed 0·75 solution is asserted to reproduce the
committed data/outputs/step1_translation/b4_b3_cascade_pairs.csv exactly
(39/39 partners, 39/39 stages).

The title embeddings are the one artefact step 1 never persists -- only their
cosine values survive, in the sim_title column of that CSV -- so they must be
recomputed here. That needs the all-mpnet-base-v2 weights and takes ~20 s; the
resulting matrices are cached in outputs/ so reruns are instant. Use --refresh
to force a rebuild.

Usage:
    python figures/supplementary/figS8_threshold_sensitivity.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import lancet_style as ls   # locked palette/typography, shared with the main-text figures

CACHE = ROOT / "data" / "outputs" / "figure_inputs" / "threshold_sensitivity_matrices.npz"
REFERENCE_PAIRS = ROOT / "data" / "outputs" / "step1_translation" / "b4_b3_cascade_pairs.csv"

# Sweep grid requested for the sensitivity panel.
TAU_MIN, TAU_MAX, TAU_STEP = 0.60, 0.90, 0.01


# ============================================================================
# DATA  (rebuilt from the pipeline's own code, then validated)
# ============================================================================
def build_matrices() -> dict:
    """Recompute the title and full-text B4xB3 cosine matrices via step 1."""
    from sklearn.metrics.pairwise import cosine_similarity

    from bayley_nlp.core import embed_items, load_model, parse_items
    from bayley_nlp.pipelines.step1_translation import (
        ITEMS_FILE_B3_ALL, ITEMS_FILE_B4, MODEL, load_theoretical_assignments,
    )

    item_to_factor, _ = load_theoretical_assignments()
    b4_all = parse_items(str(ITEMS_FILE_B4))
    b4_items = b4_all[b4_all["item_id"].isin(item_to_factor)].copy().reset_index(drop=True)
    b3_items = parse_items(str(ITEMS_FILE_B3_ALL)).reset_index(drop=True)

    model = load_model(MODEL)
    sim_fulltext = cosine_similarity(
        embed_items(b4_items, text_column="item_text", model=model),
        embed_items(b3_items, text_column="item_text", model=model),
    )
    sim_title = cosine_similarity(
        embed_items(b4_items, text_column="item_title", model=model),
        embed_items(b3_items, text_column="item_title", model=model),
    )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, sim_title=sim_title, sim_fulltext=sim_fulltext,
             b4_ids=b4_items["item_id"].to_numpy(), b3_ids=b3_items["item_id"].to_numpy())
    return {"sim_title": sim_title, "sim_fulltext": sim_fulltext,
            "b4_ids": b4_items["item_id"].to_numpy(), "b3_ids": b3_items["item_id"].to_numpy()}


def load_matrices(refresh: bool = False) -> dict:
    if refresh or not CACHE.exists():
        print("  rebuilding similarity matrices (needs all-mpnet-base-v2)...")
        return build_matrices()
    print(f"  using cached matrices: {CACHE.name}")
    z = np.load(CACHE, allow_pickle=True)
    return {k: z[k] for k in ("sim_title", "sim_fulltext", "b4_ids", "b3_ids")}


def cascade(sim_title: np.ndarray, sim_fulltext: np.ndarray, tau: float) -> dict[int, int]:
    """Run the pipeline's own cascade at threshold `tau`; return {b4_idx: b3_idx}."""
    from bayley_nlp.pipelines.step1_translation import _cascade_pairing
    pairs, _, _, _ = _cascade_pairing(sim_title, sim_fulltext, tau)
    return {i: j for (i, j, _, _) in pairs}


def validate(d: dict) -> None:
    """Fail loudly if the rebuilt 0·75 solution is not the committed one."""
    from bayley_nlp.pipelines.step1_translation import CASCADE_TITLE_THRESHOLD

    ref = pd.read_csv(REFERENCE_PAIRS)
    got = cascade(d["sim_title"], d["sim_fulltext"], CASCADE_TITLE_THRESHOLD)
    b4_ids, b3_ids = d["b4_ids"], d["b3_ids"]
    rebuilt = {str(b4_ids[i]): str(b3_ids[j]) for i, j in got.items()}

    if len(rebuilt) != len(ref):
        raise SystemExit(f"pair count differs: rebuilt {len(rebuilt)} vs committed {len(ref)}")
    bad = [r.b4_item_id for r in ref.itertuples()
           if rebuilt.get(r.b4_item_id) != r.b3_item_id]
    if bad:
        raise SystemExit(f"rebuilt pairing differs from the committed CSV for: {bad}")
    print(f"  validated: rebuilt 0·75 solution reproduces all {len(ref)} committed pairs")


# ============================================================================
# PANELS
# ============================================================================
ABOVE = ls.PAIRING_COLORS["title_locked"]    # #1B5FA8
BELOW = ls.PAIRING_COLORS["fulltext"]        # #E07B39
THRESHOLD = 0.75


def panel_a(ax, max_title: np.ndarray) -> None:
    """Histogram of the best title similarity available to each Bayley-4 item."""
    vals = np.clip(max_title, None, 1.0)          # trim float32 overshoot at 1·0
    edges = np.arange(0.35, 1.0001, 0.025)        # 0·75 falls exactly on an edge
    counts, _ = np.histogram(vals, bins=edges)

    for lo, hi, n in zip(edges[:-1], edges[1:], counts):
        if n == 0:
            continue
        ax.bar(lo, n, width=(hi - lo) * 0.92, align="edge", zorder=2,
               color=(ABOVE if lo >= THRESHOLD else BELOW),
               edgecolor="white", linewidth=0.4)

    # every individual item, so the reader sees 39 observations and not just bins
    ax.plot(vals, np.full_like(vals, -0.55), marker="|", linestyle="none",
            markersize=4.0, markeredgewidth=0.7, color="#555555",
            clip_on=False, zorder=3)

    ax.axvline(THRESHOLD, color="#111111", linestyle=(0, (4, 2)), linewidth=1.0, zorder=4)
    ax.text(THRESHOLD - 0.012, counts.max() * 0.80, "threshold 0·75",
            ha="right", va="top", fontsize=7.5, fontweight="bold", color="#111111")

    # Each count label sits over the side of the threshold it describes; the
    # white backing keeps the threshold rule from striking through the text.
    n_above, n_below = int((vals >= THRESHOLD).sum()), int((vals < THRESHOLD).sum())
    backing = dict(facecolor="white", edgecolor="none", pad=1.6)
    ax.text(0.015, 0.98, f"resolved on full text  {n_below}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=BELOW, fontweight="bold",
            bbox=backing, zorder=6)
    ax.text(0.985, 0.98, f"resolved on item title  {n_above}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=ABOVE, fontweight="bold",
            bbox=backing, zorder=6)

    # The widest empty interval in the observed data is NOT at the threshold.
    # Mark it so the panel cannot be read as a clean two-mode separation.
    srt = np.sort(vals)
    k = int(np.argmax(np.diff(srt)))
    gap_lo, gap_hi = srt[k], srt[k + 1]
    ax.annotate(
        "", xy=(gap_lo, counts.max() * 0.52), xytext=(gap_hi, counts.max() * 0.52),
        arrowprops=dict(arrowstyle="<->", color="#777777", linewidth=0.7, shrinkA=0, shrinkB=0))
    ax.text((gap_lo + gap_hi) / 2, counts.max() * 0.56,
            f"widest gap\n{ls.fmt_num(gap_lo)} to {ls.fmt_num(gap_hi)}",
            ha="center", va="bottom", fontsize=6.8, color="#777777", linespacing=1.25)

    ax.set_xlim(0.35, 1.005)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_xlabel("Highest title similarity to any Bayley-III item")
    ax.set_ylabel("Number of Bayley-4 items")
    ax.xaxis.set_major_formatter(ls.decimal_formatter(2))
    ax.yaxis.set_major_formatter(ls.decimal_formatter(None))
    ax.set_xticks(np.arange(0.4, 1.001, 0.1))
    ax.set_yticks(np.arange(0, counts.max() + 1, 5))   # counts are integers
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_b(ax, taus: np.ndarray, changed: np.ndarray) -> None:
    """Pairings that differ from the adopted 0·75 solution, across the sweep."""
    stable = taus[changed == 0]
    lo, hi = stable.min(), stable.max()
    ax.axvspan(lo, hi, color="#dce7f4", zorder=0)
    ax.text((lo + hi) / 2, changed.max() * 0.97,
            f"identical solution\n{ls.fmt_num(lo)} to {ls.fmt_num(hi)}",
            ha="center", va="top", fontsize=7.0, color="#1B5FA8",
            fontweight="bold", linespacing=1.3)

    ax.step(taus, changed, where="mid", color="#333333", linewidth=1.2, zorder=3)
    ax.plot(taus, changed, marker="o", linestyle="none", markersize=2.6,
            color="#333333", zorder=4)

    ax.axvline(THRESHOLD, color="#111111", linestyle=(0, (4, 2)), linewidth=1.0, zorder=5)
    ax.text(THRESHOLD - 0.006, changed.max() * 0.60, "adopted threshold 0·75",
            ha="right", va="center", fontsize=7.5, fontweight="bold", color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=6)

    ax.set_xlim(TAU_MIN - 0.005, TAU_MAX + 0.005)
    ax.set_ylim(-0.22, max(changed.max() * 1.16, 1.0))
    ax.set_xlabel("Title-similarity threshold")
    ax.set_ylabel("Pairings differing from the 0·75 solution")
    ax.xaxis.set_major_formatter(ls.decimal_formatter(2))
    ax.yaxis.set_major_formatter(ls.decimal_formatter(None))
    ax.set_xticks(np.arange(0.60, 0.901, 0.05))
    ax.set_yticks(np.arange(0, changed.max() + 1, 1))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)


# ============================================================================
# SAVE  (exact canvas, so the SVG really is 174 mm wide)
# ============================================================================
FIG_W_MM, FIG_H_MM = 174.0, 66.0


def save_exact(fig, stem: str, outdir: Path) -> None:
    """Save without a tight bounding box, then label the SVG in millimetres.

    ls.save_figure uses bbox_inches="tight", which re-crops the canvas to the
    drawn content and would deliver ~185 mm rather than the 174 mm column the
    journal specifies. Everything here is laid out to fit the stated canvas, so
    the figure is written at its exact size instead.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png", "svg"):
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=400, facecolor="white")
        print(f"  saved: {path}")

    # matplotlib writes the size in points; restate it in millimetres so the
    # physical width is unambiguous to a typesetter. The viewBox is untouched,
    # so the geometry is identical either way.
    svg = outdir / f"{stem}.svg"
    text = svg.read_text(encoding="utf-8")
    text = text.replace(f'width="{FIG_W_MM / 25.4 * 72:.6f}pt"', f'width="{FIG_W_MM}mm"', 1)
    text = text.replace(f'height="{FIG_H_MM / 25.4 * 72:.6f}pt"', f'height="{FIG_H_MM}mm"', 1)
    svg.write_text(text, encoding="utf-8")


# ============================================================================
# BUILD
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the embeddings instead of using the cache")
    args = ap.parse_args()

    ls.set_lancet_rcparams(base=8.0)

    d = load_matrices(refresh=args.refresh)
    validate(d)

    sim_title, sim_fulltext = d["sim_title"], d["sim_fulltext"]
    max_title = sim_title.max(axis=1)

    taus = np.round(np.arange(TAU_MIN, TAU_MAX + TAU_STEP / 2, TAU_STEP), 2)
    reference = cascade(sim_title, sim_fulltext, THRESHOLD)
    changed = np.array([
        sum(1 for i in reference if cascade(sim_title, sim_fulltext, t).get(i) != reference[i])
        for t in taus
    ])

    fig, axes = plt.subplots(1, 2, figsize=(ls.mm(FIG_W_MM), ls.mm(FIG_H_MM)))
    fig.subplots_adjust(left=0.072, right=0.992, bottom=0.155, top=0.915, wspace=0.245)

    panel_a(axes[0], max_title)
    panel_b(axes[1], taus, changed)
    ls.panel_label(axes[0], "A", x=-0.068, y=1.01)
    ls.panel_label(axes[1], "B", x=-0.068, y=1.01)

    save_exact(fig, "figS8_threshold_sensitivity", ROOT / "results" / "figures" / "supplementary")

    band = changed[(taus >= 0.70) & (taus <= 0.80)]
    print(f"\n  n items            : {len(max_title)}")
    print(f"  above threshold    : {int((max_title >= THRESHOLD).sum())}")
    print(f"  in 0·70 to 0·80    : {int(((max_title >= 0.70) & (max_title <= 0.80)).sum())}")
    print(f"  max pairings changed across 0·70 to 0·80 : {band.max()}")
    print(f"  max pairings changed across 0·60 to 0·90 : {changed.max()}")


if __name__ == "__main__":
    main()
