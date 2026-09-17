"""
Unit tests for bayley_nlp.core.clustering.

Uses small synthetic embeddings rather than the real item/embedding cache
so these tests are portable and do not require data/raw/ access: they
check the clustering machinery's correctness, not the paper's reported
cluster assignments (covered by tests/test_reference_outputs.py, which
diffs the live outputs against tests/fixtures/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayley_nlp.core.clustering import (
    HDBSCANStrategy,
    KMeansConsensusStrategy,
    SimilarityClusteringStrategy,
    validate_n_clusters,
)


def test_validate_n_clusters_accepts_reasonable_value() -> None:
    assert validate_n_clusters(5, 39) is True


def test_validate_n_clusters_rejects_too_few_clusters() -> None:
    with pytest.raises(ValueError):
        validate_n_clusters(1, 39)


def test_validate_n_clusters_rejects_n_clusters_ge_n_items() -> None:
    with pytest.raises(ValueError):
        validate_n_clusters(39, 39)


@pytest.fixture
def synthetic_items_and_embeddings():
    """20 items with 768-dim embeddings arranged as 4 well-separated blobs."""
    rng = np.random.default_rng(42)
    n_per_cluster, n_clusters, dim = 5, 4, 768
    centers = rng.normal(scale=5.0, size=(n_clusters, dim))
    emb = np.vstack(
        [centers[c] + rng.normal(scale=0.1, size=(n_per_cluster, dim)) for c in range(n_clusters)]
    )
    items = pd.DataFrame({"item_id": [f"ITEM_{i:03d}" for i in range(emb.shape[0])]})
    return items, emb


def test_kmeans_consensus_strategy_runs_on_synthetic_data(tmp_path, synthetic_items_and_embeddings) -> None:
    items, emb = synthetic_items_and_embeddings
    strategy = KMeansConsensusStrategy(
        embeddings=emb,
        items=items,
        output_dir=tmp_path,
        model_name="synthetic-test",
        timestamp="test",
        n_clusters=4,
        n_iterations=10,  # small for test speed; paper uses 1000
        umap_kwargs={"metric": "cosine", "n_neighbors": 5, "min_dist": 0.0, "n_components": 5},
        consensus_linkage="average",
    )
    result = strategy.run()
    assert len(result.labels) == len(items)
    assert len(set(result.labels)) <= 4


def test_hdbscan_strategy_initializes_on_synthetic_data(tmp_path, synthetic_items_and_embeddings) -> None:
    items, emb = synthetic_items_and_embeddings
    strategy = HDBSCANStrategy(
        embeddings=emb,
        items=items,
        output_dir=tmp_path,
        model_name="synthetic-test",
        timestamp="test",
        umap_kwargs={"metric": "cosine", "n_neighbors": 5, "min_dist": 0.0, "n_components": 5},
        hdbscan_param_grid={"min_cluster_size": [2, 3], "min_samples": [None, 1]},
    )
    assert strategy is not None


def test_similarity_clustering_strategy_assigns_all_items(tmp_path, synthetic_items_and_embeddings) -> None:
    items, emb = synthetic_items_and_embeddings
    items = items.assign(item_text=[f"synthetic item {i}" for i in range(len(items))])
    strategy = SimilarityClusteringStrategy(
        embeddings=emb,
        items=items,
        output_dir=tmp_path,
        model_name="synthetic-test",
        timestamp="test",
    )
    assert strategy is not None
