from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np
import pandas as pd

from .dimensionality_reduction import get_reducer, reduce_dimensions, CORE_METHODS


@dataclass
class HDBSCANResult:
    labels: np.ndarray
    probabilities: np.ndarray
    params: Dict[str, Optional[int]]
    diagnostics: Dict[str, object]


def compute_dbcv(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """Compute DBCV index using the hdbscan validity implementation.

    Returns NaN if fewer than two clusters are present.
    """
    try:
        from hdbscan.validity import validity_index
    except Exception as e:
        raise RuntimeError("Install hdbscan: pip install hdbscan") from e

    n_clusters = len({c for c in np.unique(labels) if c != -1})
    if n_clusters < 2:
        return float("nan")
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int32)
    return float(validity_index(X64, labs, metric=metric))


def masked_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """Silhouette on non-noise points (labels != -1). Returns NaN if < 2 clusters."""
    from sklearn.metrics import silhouette_score

    mask = labels != -1
    if mask.sum() == 0:
        return float("nan")
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labels[mask], metric=metric))
    except Exception:
        return float("nan")


def umap_reduce(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_components: int = 15,
    metric: str = "cosine",
    random_state: int = 10,
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction (McInnes et al., 2018).
    
    This is a backward-compatible wrapper around the dimensionality_reduction module.
    For new code, prefer using get_reducer() or reduce_dimensions() directly.
    """
    reducer = get_reducer(
        "umap",
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(X)


def _apply_dimensionality_reduction(
    X: np.ndarray,
    method: str = "umap",
    random_state: int = 10,
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Apply dimensionality reduction using the specified method.
    
    Args:
        X: Input embeddings (n_samples, n_features)
        method: DR method name (umap, tsne, pca, isomap, spectral, mds)
        random_state: Random seed for reproducibility
        verbose: Show progress
        **kwargs: Method-specific parameters (n_neighbors, n_components, etc.)
        
    Returns:
        Reduced embeddings
    """
    reducer = get_reducer(
        method,
        random_state=random_state,
        verbose=verbose,
        **kwargs
    )
    return reducer.fit_transform(X)


def umap_hdbscan_auto(
    X: np.ndarray,
    umap_kwargs: Optional[Dict] = None,
    hdbscan_param_grid: Optional[Dict[str, Iterable]] = None,
    random_state: int = 10,
    use_dr: bool = True,
    dr_method: str = "umap",
) -> HDBSCANResult:
    """Run dimensionality reduction + HDBSCAN with model selection via DBCV.

    Args:
        X: Input embeddings (n_samples, n_features)
        umap_kwargs: Parameters for dimensionality reduction (n_neighbors, n_components, etc.)
        hdbscan_param_grid: Grid of HDBSCAN parameters to search
        random_state: Random seed for reproducibility
        use_dr: Whether to apply dimensionality reduction before clustering
        dr_method: Dimensionality reduction method (umap, tsne, pca, isomap, spectral, mds)
        
    Returns:
        HDBSCANResult with labels/probabilities and diagnostics.
    """
    n, _ = X.shape

    if umap_kwargs is None:
        umap_kwargs = {
            "metric": "cosine",
            "n_neighbors": 15,
            "min_dist": 0.0,
            "n_components": 15,
        }

    if hdbscan_param_grid is None:
        sqrt_n = max(2, int(round(math.sqrt(n))))
        mcs = sorted({sqrt_n, 10, 15, 25, 50})
        ms = [None, 1, 5, 10]
        hdbscan_param_grid = {"min_cluster_size": mcs, "min_samples": ms}

    # Dimensionality reduction
    if use_dr:
        Xr = _apply_dimensionality_reduction(
            X,
            method=dr_method,
            n_neighbors=int(umap_kwargs.get("n_neighbors", 15)),
            min_dist=float(umap_kwargs.get("min_dist", 0.0)),
            n_components=int(umap_kwargs.get("n_components", 15)),
            metric=str(umap_kwargs.get("metric", "cosine")),
            random_state=random_state,
            verbose=False,
        )
    else:
        Xr = X

    # Grid search HDBSCAN by DBCV, tie-break on silhouette then noise
    from hdbscan import HDBSCAN

    results: List[Dict[str, object]] = []
    for mcs in hdbscan_param_grid.get("min_cluster_size", []):
        for ms in hdbscan_param_grid.get("min_samples", []):
            clusterer = HDBSCAN(
                metric="euclidean",
                min_cluster_size=int(mcs),
                min_samples=None if ms is None else int(ms),
                prediction_data=False,
                core_dist_n_jobs=1,
            )
            clusterer.fit(Xr)
            labels = clusterer.labels_.astype(int)
            probs = clusterer.probabilities_.astype(float)

            dbcv = compute_dbcv(Xr, labels, metric="euclidean")
            sil = masked_silhouette(Xr, labels, metric="euclidean")
            noise_rate = float(np.mean(labels == -1))
            sizes = (
                pd.Series(labels)
                .loc[lambda s: s != -1]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            results.append(
                {
                    "min_cluster_size": int(mcs),
                    "min_samples": (None if ms is None else int(ms)),
                    "labels": labels,
                    "probabilities": probs,
                    "dbcv": dbcv,
                    "silhouette": sil,
                    "noise_rate": noise_rate,
                    "cluster_sizes": sizes,
                }
            )

    def _score_tuple(r: Dict[str, object]) -> Tuple[float, float, float]:
        dbcv = r["dbcv"]
        sil = r["silhouette"]
        noise = r["noise_rate"]
        dbcv_s = -1e9 if (dbcv is None or (isinstance(dbcv, float) and math.isnan(dbcv))) else float(dbcv)
        sil_s = -1e9 if (sil is None or (isinstance(sil, float) and math.isnan(sil))) else float(sil)
        return (dbcv_s, sil_s, -float(noise))

    best = max(results, key=_score_tuple)

    # Post-process: ensure no point is labeled as noise (-1)
    labels_best = best["labels"].copy()
    probs_best = best["probabilities"].copy()
    if np.any(labels_best == -1):
        mask_nn = labels_best != -1
        for i in np.where(~mask_nn)[0]:
            d = np.linalg.norm(Xr[mask_nn] - Xr[i], axis=1)
            j_rel = int(np.argmin(d))
            abs_indices = np.flatnonzero(mask_nn)
            j_abs = int(abs_indices[j_rel])
            labels_best[i] = int(labels_best[j_abs])
            probs_best[i] = float(probs_best[j_abs])

    noise_rate_final = float(np.mean(labels_best == -1))
    sizes_final = (
        pd.Series(labels_best)
        .loc[lambda s: s != -1]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    diagnostics = {
        "dbcv": best["dbcv"],
        "silhouette": best["silhouette"],
        "noise_rate": noise_rate_final,
        "cluster_sizes": sizes_final,
        "grid_results": [
            {
                "min_cluster_size": r["min_cluster_size"],
                "min_samples": r["min_samples"],
                "dbcv": r["dbcv"],
                "silhouette": r["silhouette"],
                "noise_rate": r["noise_rate"],
            }
            for r in results
        ],
        "used_dr": bool(use_dr),
        "dr_method": dr_method if use_dr else None,
        "umap_kwargs": dict(umap_kwargs),
    }

    return HDBSCANResult(
        labels=labels_best,
        probabilities=probs_best,
        params={
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": best["min_samples"],
        },
        diagnostics=diagnostics,
    )

