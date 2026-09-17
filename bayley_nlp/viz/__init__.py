"""Visualization and reporting utilities."""

from .plotting import (
    plot_umap_2d_scatter,
    plot_cluster_size_hist,
    plot_dbcv_vs_mcs,
    plot_heatmap,
    compute_tsne_coords,
    compute_coords,
    plot_tsne_clusters,
    plot_scree,
    plot_structure_summary,
)

# Note: reporting.py is in root bayley_nlp/, will be moved/imported separately

__all__ = [
    "plot_umap_2d_scatter",
    "plot_cluster_size_hist",
    "plot_dbcv_vs_mcs",
    "plot_heatmap",
    "compute_tsne_coords",
    "compute_coords",
    "plot_tsne_clusters",
    "plot_scree",
    "plot_structure_summary",
]
