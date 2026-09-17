"""
_validation.py — data + drawers for the unsupervised-validation figure (Fig 5).

Computes the theoretical × detected contingency (confusion) matrices for the
Bayley-4 (39 items, Step 2) and Bayley-III (91 items, Step 3) item sets, plus
ARI/NMI, reusing the exact assignment CSVs behind the committed figures. Also
provides a clean matplotlib confusion-heatmap drawer.

Row order (theoretical) and column order (detected) are the canonical
[WM, FS, GDPS, ATT, HOP] so the diagonal is meaningful.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import lancet_style as ls

ORDER = ["WM", "FS", "GDPS", "ATT", "HOP"]
_IDX = {f: i for i, f in enumerate(ORDER)}

_NAME_MAP = {
    "working memory": "WM", "w": "WM",
    "flexibility shift": "FS", "f": "FS",
    "goal directed problem solving": "GDPS", "g": "GDPS",
    "attention": "ATT", "a": "ATT",
    "higher order processing": "HOP", "higher order processes": "HOP", "h": "HOP",
}


def _short(name) -> str | None:
    key = str(name).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return _NAME_MAP.get(key)


def _count_matrix(theo_labels, det_labels):
    """5×5 contingency (rows=theoretical, cols=detected) from short-code lists."""
    M = np.zeros((5, 5), dtype=int)
    for t, d in zip(theo_labels, det_labels):
        if t in _IDX and d in _IDX:
            M[_IDX[t], _IDX[d]] += 1
    return M


def _metrics(theo_labels, det_labels):
    """ARI/NMI + n on one-label-per-item partitions."""
    pairs = [(t, d) for t, d in zip(theo_labels, det_labels)
             if t in _IDX and d in _IDX]
    if not pairs:
        return np.nan, np.nan, 0
    ti = [_IDX[t] for t, _ in pairs]
    di = [_IDX[d] for _, d in pairs]
    return (adjusted_rand_score(ti, di),
            normalized_mutual_info_score(ti, di), len(pairs))


def _matrix(theo_labels, det_labels):
    """One-label-per-item: matrix + ARI/NMI (used by the B3 loader)."""
    M = _count_matrix(theo_labels, det_labels)
    ari, nmi, n = _metrics(theo_labels, det_labels)
    return M, ari, nmi, n


# ── Bayley-4 (Step 2) ────────────────────────────────────────────────────────
def load_b4(root: Path):
    p = (root / "data/outputs/step2_b4_validation"
         / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
         / "05_model_comparison" / "item_comparison_theoretical.csv")
    df = pd.read_csv(p)

    exp_theo, exp_det = [], []       # multi-factor expanded → confusion matrix
    prim_theo, prim_det = [], []     # primary factor per item → ARI/NMI
    for _, r in df.iterrows():
        det = _short(r["detected_factor"])
        refs = [x.strip() for x in str(r["reference_factor"]).split(",")]
        if not refs or refs[0] == "unassigned":
            continue
        pt = _short(refs[0])
        if pt and det:
            prim_theo.append(pt); prim_det.append(det)
        for rf in refs:                       # expand multi-factor items
            t = _short(rf)
            if t and det:
                exp_theo.append(t); exp_det.append(det)

    M = _count_matrix(exp_theo, exp_det)
    ari, nmi, n = _metrics(prim_theo, prim_det)
    return M, ari, nmi, n


# ── Bayley-III (Step 3) ──────────────────────────────────────────────────────
def load_b3(root: Path):
    run = root / "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"
    theo_base = root / "data/outputs/step1_translation"

    clusters = pd.read_csv(run / "03_clustering" / "cluster_assignments.csv")
    labels = pd.read_csv(run / "04_labeling" / "cluster_concept_labels.csv")
    cid2short = {int(r["cluster_id"]): _short(r["top_concept"]) for _, r in labels.iterrows()}
    det_map = {r["item_id"]: cid2short.get(int(r["cluster"])) for _, r in clusters.iterrows()}

    # Canonical translated model (cascade pairing + centroid assignment + expert
    # review of the UC-flagged items), read from results/tables/
    # bayley3_item_domain_assignments.csv — shared with fig2 so the figures stay
    # consistent. Distribution: ATT 22/WM 19/GDPS 23/FS 20/HOP 7.
    import _embedding as emb
    theo_map = emb.translated_short_map(root)

    items = sorted(set(theo_map) & set(det_map))
    theo = [theo_map[i] for i in items]
    det = [det_map[i] for i in items]
    return _matrix(theo, det)


# ── Confusion-matrix heatmap drawer ──────────────────────────────────────────
def draw_confusion(ax, M, row_axis="Theoretical factor", col_axis="Detected cluster",
                   cmap="Blues"):
    n = M.shape[0]
    vmax = M.max() if M.max() > 0 else 1
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v == 0:
                continue
            dark = v > 0.6 * vmax
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if dark else "#1a1a1a")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER); ax.set_yticklabels(ORDER)
    for tick, f in zip(ax.get_xticklabels(), ORDER):
        tick.set_color(ls.FACTOR_COLORS[f]); tick.set_fontweight("bold")
    for tick, f in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(ls.FACTOR_COLORS[f]); tick.set_fontweight("bold")
    ax.set_xlabel(col_axis, fontsize=8.5)
    ax.set_ylabel(row_axis, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name, loader in [("Bayley-4", load_b4), ("Bayley-III", load_b3)]:
        M, ari, nmi, n = loader(root)
        print(f"\n{name}: n={n}  ARI={ari:.3f}  NMI={nmi:.3f}")
        print("rows=theoretical, cols=detected, order", ORDER)
        print(M)
