from typing import Optional, Sequence, Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bayley_nlp.utils import ensure_dir

def plot_umap_2d_scatter(
    X2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    outpath: str,
    item_ids: Optional[Sequence[str]] = None,
) -> None:
    """Scatter plot with clusters colored, noise in gray.
    Optionally annotate with item IDs and add legend mapping clusters to colors.
    """
    ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(9, 8))

    # Determine palette excluding noise
    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    palette = sns.color_palette("tab10", n_colors=max(1, len(cluster_ids)))
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}
    colors = [color_map.get(int(lab), (0.6, 0.6, 0.6)) for lab in labels]  # gray for -1

    # Scatter plot
    plt.scatter(X2d[:, 0], X2d[:, 1], c=colors, s=30, alpha=0.9, edgecolors="none")

    # Annotate points if item_ids are provided
    if item_ids is not None and len(item_ids) == X2d.shape[0]:
        for (x, y, lab, name) in zip(X2d[:, 0], X2d[:, 1], labels, item_ids):
            plt.annotate(str(name), (x, y), fontsize=7, alpha=0.75, ha="center", va="center")

    # Build legend handles
    import matplotlib.patches as mpatches
    handles = []
    for cid in cluster_ids:
        handles.append(mpatches.Patch(color=color_map[cid], label=f"Cluster {cid+1}"))
    if -1 in labels:
        handles.append(mpatches.Patch(color=(0.6, 0.6, 0.6), label="Noise"))

    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_cluster_size_hist(labels: np.ndarray, outpath: str) -> None:
    ensure_dir(os.path.dirname(outpath))
    lab = pd.Series(labels)
    counts = lab[lab != -1].value_counts().sort_index()
    counts =+ 1
    plt.figure(figsize=(7, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="steelblue")
    plt.xlabel("Cluster ID")
    plt.ylabel("Size")
    plt.title("Cluster Sizes")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_dbcv_vs_mcs(grid_results: List[Dict[str, object]], outpath: str) -> None:
    """Plot best DBCV per min_cluster_size across min_samples."""
    df = pd.DataFrame(grid_results)
    agg = (
        df.groupby("min_cluster_size").apply(lambda g: g.loc[g["dbcv"].fillna(-1e9).idxmax()]).reset_index(drop=True)
    )
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=agg, x="min_cluster_size", y="dbcv", marker="o")
    plt.xlabel("min_cluster_size")
    plt.ylabel("DBCV (higher is better)")
    plt.title("DBCV vs min_cluster_size (best min_samples)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_heatmap(S: np.ndarray, item_ids: Sequence[str], outpath: str, figsize=(10, 8)) -> None:
    # Apply a clean theme
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=figsize)
    # Heatmap with clearer colorbar label and square cells for readability
    hm = sns.heatmap(
        S,
        cmap="viridis",
        xticklabels=item_ids,
        yticklabels=item_ids,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    ax.set_title("Cosine Similarity Matrix")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def compute_tsne_coords(S: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        random_state=random_state,
        init="random",
        perplexity=perplexity,
    )
    coords = tsne.fit_transform(1 - S)  # distance = 1 - similarity
    return coords


def compute_coords(
    S: np.ndarray,
    emb: np.ndarray,
    method: str = "tsne",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D coordinates using various DR methods.

    - tsne: t-SNE on precomputed distances (1 - S)
    - pca: PCA on embedding features
    - mds: Metric MDS on precomputed distances (1 - S)
    - isomap: Isomap on precomputed distances (1 - S)
    - spectral: SpectralEmbedding on precomputed affinity S
    - umap: UMAP on embedding features (if available)
    """

    method = (method or "tsne").lower()
    rng = random_state
    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            random_state=rng,
            init="random",
            perplexity=perplexity,
        )
        return tsne.fit_transform(1 - S)
    elif method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=rng).fit_transform(emb)
    elif method == "mds":
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=rng)
        return mds.fit_transform(1 - S)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        iso = Isomap(n_components=2, metric="precomputed")
        return iso.fit_transform(1 - S)
    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding
        A = 0.5 * (S + S.T)
        A = np.where(A < 0, 0.0, A)
        np.fill_diagonal(A, 1.0)
        se = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=rng)
        return se.fit_transform(A)
    elif method == "umap":
        try:
            import umap 
        except Exception as e:
            raise RuntimeError("UMAP not installed. pip install umap-learn") from e
        um = umap.UMAP(n_components=2, random_state=rng)
        return um.fit_transform(emb)
    else:
        raise ValueError(f"Unknown DR method: {method}")


def plot_tsne_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    item_ids: Sequence[str],
    title: str,
    outpath: str,
    figsize=(8, 6),
) -> None:
    """
    Scatter plot of 2D coords colored by cluster labels with informative legend.
    Legend entries are shown as "Cluster <n>" with a legend title "Color: Cluster".
    """
    import matplotlib.patheffects as pe

    sns.set_theme(style="whitegrid", context="notebook")

    # Prepare data
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels.astype(int),
        "item_id": list(item_ids),
    })
    unique_clusters = sorted(df["cluster"].unique().tolist())
    n_colors = max(1, len(unique_clusters))

    # Choose a palette that scales to many clusters
    if n_colors <= 10:
        palette = sns.color_palette("tab10", n_colors)
    elif n_colors <= 20:
        palette = sns.color_palette("tab20", n_colors)
    else:
        palette = sns.color_palette("hls", n_colors)

    # Map cluster id -> name and categorical order for clean legend labels
    cluster_names = [f"Cluster {c}" for c in unique_clusters]
    mapping = {c: f"Cluster {c}" for c in unique_clusters}
    df["cluster_name"] = pd.Categorical(df["cluster"].map(mapping), categories=cluster_names, ordered=True)

    fig, ax = plt.subplots(figsize=figsize)
    # Scatter with nice aesthetics
    sc = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="cluster_name",
        palette=palette,
        s=70,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        ax=ax,
        legend=True,
    )

    # Annotate each point with item_id, with white stroke for readability
    for _, row in df.iterrows():
        txt = ax.text(row["x"], row["y"], str(row["item_id"]), fontsize=8, zorder=3)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Color: Cluster", frameon=True, labelspacing=0.4, borderpad=0.6)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_scree(
    eigenvalues: np.ndarray,
    random_eigenvalues: Optional[np.ndarray],
    outpath: str,
    chosen_k: Optional[int] = None,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    xs = np.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, eigenvalues, marker="o", linewidth=2, markersize=5, label="Observed eigenvalues")
    if random_eigenvalues is not None:
        ax.plot(xs, random_eigenvalues, marker="o", linestyle="--", linewidth=1.5, markersize=4, label="Random mean eigenvalues")
    if chosen_k is not None:
        ax.axvline(chosen_k, color="crimson", linestyle="--", alpha=0.7, label=f"Chosen k = {chosen_k}")
    ax.set_xlabel("Component number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Parallel Analysis Scree")
    ax.legend(frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_structure_summary(counts: Dict[str, int], outpath: str) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    labels = list(counts.keys())
    values = [int(counts[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette("Set2", n_colors=len(labels))
    sns.barplot(x=labels, y=values, palette=palette, ax=ax, edgecolor="white")
    # Add value labels above bars with some headroom
    ymax = max(values) if values else 1
    ax.set_ylim(0, ymax * 1.15)
    for i, v in enumerate(values):
        ax.text(i, v + ymax * 0.02, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Detected Structure: Factors and Clusters")
    ax.tick_params(axis="x", rotation=20)
    sns.despine()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
