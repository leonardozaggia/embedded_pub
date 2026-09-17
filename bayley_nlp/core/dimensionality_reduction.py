"""
Dimensionality Reduction Module

Provides a unified interface for multiple dimensionality reduction algorithms,
enabling easy algorithm comparison and swapping in the clustering pipeline.

Supported algorithms:
- UMAP: Uniform Manifold Approximation and Projection
- t-SNE: t-Distributed Stochastic Neighbor Embedding
- PCA: Principal Component Analysis
- Isomap: Isometric Mapping
- Spectral: Spectral Embedding
- MDS: Multidimensional Scaling
- TriMAP: Large-scale Dimensionality Reduction (optional)
- PACMAP: Pairwise Controlled Manifold Approximation (optional)

Usage:
    from bayley_nlp.core.dimensionality_reduction import get_reducer, reduce_dimensions
    
    # Quick reduction with defaults
    X_reduced = reduce_dimensions(embeddings, method="umap", n_components=15)
    
    # With custom parameters
    reducer = get_reducer("tsne", n_components=2, perplexity=50)
    X_2d = reducer.fit_transform(embeddings)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import yaml
from tqdm import tqdm


# ============================================================================
# Configuration Loading
# ============================================================================

def load_dr_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load dimensionality reduction configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Dictionary of configuration parameters by algorithm.
    """
    if config_path is None:
        # Default config location relative to this file
        config_path = Path(__file__).resolve().parents[2] / "config" / "dimensionality_reduction.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        warnings.warn(f"DR config not found at {config_path}, using hardcoded defaults")
        return _get_fallback_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def _get_fallback_config() -> Dict[str, Any]:
    """Return hardcoded fallback configuration if YAML not found."""
    return {
        "defaults": {"random_state": 42, "n_components_cluster": 15, "n_components_viz": 2},
        "umap": {"n_neighbors": 10, "min_dist": 0.0, "metric": "cosine"},
        "tsne": {"perplexity": 30, "learning_rate": "auto", "n_iter": 1000, "metric": "cosine"},
        "pca": {"svd_solver": "auto", "whiten": False},
        "isomap": {"n_neighbors": 10, "metric": "cosine"},
        "spectral": {"n_neighbors": 10, "affinity": "nearest_neighbors"},
        "mds": {"metric": True, "n_init": 4, "max_iter": 300},
    }


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReductionResult:
    """Result from dimensionality reduction."""
    embeddings: np.ndarray  # Reduced embeddings (n_samples, n_components)
    method: str             # Algorithm name
    n_components: int       # Number of output dimensions
    params: Dict[str, Any]  # Parameters used
    metadata: Dict[str, Any] = field(default_factory=dict)  # Algorithm-specific info
    
    def save(self, path: Union[str, Path]) -> None:
        """Save reduced embeddings to .npy file."""
        np.save(path, self.embeddings)
    
    @property
    def shape(self) -> tuple:
        return self.embeddings.shape


# ============================================================================
# Base Class
# ============================================================================

class DimensionalityReducer(ABC):
    """
    Abstract base class for dimensionality reduction algorithms.
    
    All reducers implement:
    - fit_transform(X): Reduce dimensions of input data
    - get_params(): Return current parameters
    """
    
    name: str = "base"
    supports_high_dim: bool = True  # Can output n_components > 3
    supports_progress: bool = False  # Has built-in progress tracking
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize reducer.
        
        Args:
            n_components: Number of output dimensions.
            random_state: Random seed for reproducibility.
            verbose: Show progress bars for long-running algorithms.
            **kwargs: Algorithm-specific parameters.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.params = kwargs
        self._fitted = False
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce dimensions of input data.
        
        Args:
            X: Input data (n_samples, n_features).
            
        Returns:
            Reduced data (n_samples, n_components).
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Return all parameters used for reduction."""
        return {
            "method": self.name,
            "n_components": self.n_components,
            "random_state": self.random_state,
            **self.params
        }
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"


# ============================================================================
# Concrete Implementations
# ============================================================================

class UMAPReducer(DimensionalityReducer):
    """
    UMAP - Uniform Manifold Approximation and Projection.
    
    Preserves both local and global structure. Fast on large datasets.
    Reference: McInnes et al. (2018) https://arxiv.org/abs/1802.03426
    """
    
    name = "umap"
    supports_high_dim = True
    supports_progress = True
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        # Suppress n_jobs warning when using random_state
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='n_jobs value.*overridden', module='umap')
            
            if self.verbose:
                print(f"  Running UMAP: n_components={self.n_components}, "
                      f"n_neighbors={self.n_neighbors}, min_dist={self.min_dist}")
            
            reducer = UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                verbose=self.verbose,
                **{k: v for k, v in self.params.items() if k not in ['n_jobs', 'low_memory']}
            )
            
            result = reducer.fit_transform(X)
            self._fitted = True
            return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
        })
        return base


class TSNEReducer(DimensionalityReducer):
    """
    t-SNE - t-Distributed Stochastic Neighbor Embedding.
    
    Excellent for 2D visualization, reveals cluster structure.
    Warning: Slow on large datasets, supports n_components <= 3.
    Reference: van der Maaten & Hinton (2008)
    """
    
    name = "tsne"
    supports_high_dim = False  # Only supports n_components <= 3
    supports_progress = True
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        metric: str = "cosine",
        init: str = "pca",
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        if n_components > 3:
            raise ValueError(f"t-SNE only supports n_components <= 3, got {n_components}")
        
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.init = init
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        
        if self.verbose:
            print(f"  Running t-SNE: n_components={self.n_components}, "
                  f"perplexity={self.perplexity}, n_iter={self.n_iter}")
            print("  (t-SNE may take several minutes for large datasets...)")
        
        reducer = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.n_iter,  # sklearn >= 1.2 uses max_iter instead of n_iter
            metric=self.metric,
            init=self.init,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            **self.params
        )
        
        result = reducer.fit_transform(X)
        self._fitted = True
        return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "metric": self.metric,
            "init": self.init,
        })
        return base


class PCAReducer(DimensionalityReducer):
    """
    PCA - Principal Component Analysis.
    
    Fast, linear, deterministic. May not capture nonlinear structure.
    """
    
    name = "pca"
    supports_high_dim = True
    supports_progress = False  # PCA is very fast
    
    def __init__(
        self,
        n_components: int = 2,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.svd_solver = svd_solver
        self.whiten = whiten
        self._explained_variance_ratio = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        
        if self.verbose:
            print(f"  Running PCA: n_components={self.n_components}")
        
        reducer = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
            **self.params
        )
        
        result = reducer.fit_transform(X)
        self._explained_variance_ratio = reducer.explained_variance_ratio_
        self._fitted = True
        
        if self.verbose:
            total_var = sum(self._explained_variance_ratio)
            print(f"  Explained variance: {total_var:.2%}")
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "svd_solver": self.svd_solver,
            "whiten": self.whiten,
        })
        if self._explained_variance_ratio is not None:
            base["explained_variance_ratio"] = self._explained_variance_ratio.tolist()
        return base


class IsomapReducer(DimensionalityReducer):
    """
    Isomap - Isometric Mapping.
    
    Uses geodesic distances for manifold learning.
    Warning: Memory-intensive for large datasets.
    Reference: Tenenbaum et al. (2000)
    """
    
    name = "isomap"
    supports_high_dim = True
    supports_progress = False
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        metric: str = "cosine",
        eigen_solver: str = "auto",
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.eigen_solver = eigen_solver
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import Isomap
        
        if self.verbose:
            print(f"  Running Isomap: n_components={self.n_components}, "
                  f"n_neighbors={self.n_neighbors}")
        
        # For cosine metric, we need to precompute the distance matrix
        if self.metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            if self.verbose:
                print("  Computing cosine distance matrix...")
            dist_matrix = cosine_distances(X)
            reducer = Isomap(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                metric="precomputed",
                eigen_solver=self.eigen_solver,
                **self.params
            )
            result = reducer.fit_transform(dist_matrix)
        else:
            reducer = Isomap(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                eigen_solver=self.eigen_solver,
                **self.params
            )
            result = reducer.fit_transform(X)
        
        self._fitted = True
        return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "eigen_solver": self.eigen_solver,
        })
        return base


class SpectralReducer(DimensionalityReducer):
    """
    Spectral Embedding.
    
    Graph-based reduction, good when data has cluster structure.
    """
    
    name = "spectral"
    supports_high_dim = True
    supports_progress = False
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        affinity: str = "nearest_neighbors",
        gamma: Optional[float] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import SpectralEmbedding
        
        if self.verbose:
            print(f"  Running Spectral Embedding: n_components={self.n_components}, "
                  f"n_neighbors={self.n_neighbors}")
        
        reducer = SpectralEmbedding(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            affinity=self.affinity,
            gamma=self.gamma,
            random_state=self.random_state,
            **self.params
        )
        
        result = reducer.fit_transform(X)
        self._fitted = True
        return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "n_neighbors": self.n_neighbors,
            "affinity": self.affinity,
            "gamma": self.gamma,
        })
        return base


class MDSReducer(DimensionalityReducer):
    """
    MDS - Multidimensional Scaling.
    
    Preserves pairwise distances.
    Warning: Slow on large datasets (O(n³) complexity).
    """
    
    name = "mds"
    supports_high_dim = True
    supports_progress = True
    
    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.metric_mds = metric  # Renamed to avoid confusion with distance metric
        self.n_init = n_init
        self.max_iter = max_iter
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import MDS
        
        if self.verbose:
            print(f"  Running MDS: n_components={self.n_components}, "
                  f"metric={self.metric_mds}, n_init={self.n_init}")
            print("  (MDS may take several minutes for large datasets...)")
        
        reducer = MDS(
            n_components=self.n_components,
            metric=self.metric_mds,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            **self.params
        )
        
        result = reducer.fit_transform(X)
        self._fitted = True
        return result
    
    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base.update({
            "metric": self.metric_mds,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
        })
        return base


# ============================================================================
# Optional Advanced Reducers (require additional packages)
# ============================================================================

class TriMAPReducer(DimensionalityReducer):
    """
    TriMAP - Large-scale Dimensionality Reduction Using Triplets.
    
    Alternative to t-SNE/UMAP, designed for very large datasets.
    Reference: Amid & Warmuth (2019)
    Requires: pip install trimap
    """
    
    name = "trimap"
    supports_high_dim = True
    supports_progress = True
    
    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 10,
        n_outliers: int = 5,
        n_random: int = 5,
        weight_adj: float = 500.0,
        n_iters: int = 400,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.weight_adj = weight_adj
        self.n_iters = n_iters
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import trimap
        except ImportError:
            raise ImportError("TriMAP not installed. Run: pip install trimap")
        
        if self.verbose:
            print(f"  Running TriMAP: n_components={self.n_components}, "
                  f"n_iters={self.n_iters}")
        
        reducer = trimap.TRIMAP(
            n_dims=self.n_components,
            n_inliers=self.n_inliers,
            n_outliers=self.n_outliers,
            n_random=self.n_random,
            weight_adj=self.weight_adj,
            n_iters=self.n_iters,
            verbose=self.verbose,
        )
        
        result = reducer.fit_transform(X)
        self._fitted = True
        return result


class PACMAPReducer(DimensionalityReducer):
    """
    PACMAP - Pairwise Controlled Manifold Approximation.
    
    Balances local and global structure preservation.
    Reference: Wang et al. (2021)
    Requires: pip install pacmap
    """
    
    name = "pacmap"
    supports_high_dim = True
    supports_progress = True
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        num_iters: int = 450,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(n_components, random_state, verbose, **kwargs)
        self.n_neighbors = n_neighbors
        self.mn_ratio = mn_ratio
        self.fp_ratio = fp_ratio
        self.num_iters = num_iters
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import pacmap
        except ImportError:
            raise ImportError("PACMAP not installed. Run: pip install pacmap")
        
        if self.verbose:
            print(f"  Running PACMAP: n_components={self.n_components}, "
                  f"n_neighbors={self.n_neighbors}")
        
        reducer = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.mn_ratio,
            FP_ratio=self.fp_ratio,
            num_iters=self.num_iters,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        
        result = reducer.fit_transform(X)
        self._fitted = True
        return result


# ============================================================================
# Factory Functions
# ============================================================================

# Registry of available reducers
REDUCER_REGISTRY: Dict[str, Type[DimensionalityReducer]] = {
    "umap": UMAPReducer,
    "tsne": TSNEReducer,
    "pca": PCAReducer,
    "isomap": IsomapReducer,
    "spectral": SpectralReducer,
    "mds": MDSReducer,
    "trimap": TriMAPReducer,
    "pacmap": PACMAPReducer,
}

# Core reducers (always available)
CORE_METHODS = ["umap", "tsne", "pca", "isomap", "spectral", "mds"]

# Optional reducers (require extra packages)
OPTIONAL_METHODS = ["trimap", "pacmap"]


def list_available_methods(include_optional: bool = True) -> List[str]:
    """
    List available dimensionality reduction methods.
    
    Args:
        include_optional: Include methods that require extra packages.
        
    Returns:
        List of method names.
    """
    methods = CORE_METHODS.copy()
    
    if include_optional:
        # Check if optional packages are available
        for method in OPTIONAL_METHODS:
            try:
                if method == "trimap":
                    import trimap
                elif method == "pacmap":
                    import pacmap
                methods.append(method)
            except ImportError:
                pass
    
    return methods


def get_reducer(
    method: str,
    n_components: int = 2,
    random_state: Optional[int] = 42,
    verbose: bool = True,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> DimensionalityReducer:
    """
    Factory function to get a dimensionality reducer.
    
    Args:
        method: Algorithm name (e.g., "umap", "tsne", "pca").
        n_components: Number of output dimensions.
        random_state: Random seed for reproducibility.
        verbose: Show progress for long-running algorithms.
        config_path: Path to config YAML. If None, uses default.
        **kwargs: Override config parameters.
        
    Returns:
        Initialized DimensionalityReducer instance.
        
    Example:
        >>> reducer = get_reducer("umap", n_components=15)
        >>> X_reduced = reducer.fit_transform(embeddings)
    """
    method = method.lower()
    
    if method not in REDUCER_REGISTRY:
        available = list_available_methods()
        raise ValueError(
            f"Unknown method '{method}'. Available: {available}"
        )
    
    # Load config defaults
    config = load_dr_config(config_path)
    method_config = config.get(method, {})
    defaults_config = config.get("defaults", {})
    
    # Merge: kwargs > method_config > defaults
    if random_state is None:
        random_state = defaults_config.get("random_state", 42)
    
    merged_kwargs = {**method_config, **kwargs}
    
    reducer_class = REDUCER_REGISTRY[method]
    return reducer_class(
        n_components=n_components,
        random_state=random_state,
        verbose=verbose,
        **merged_kwargs
    )


def reduce_dimensions(
    X: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    random_state: Optional[int] = 42,
    verbose: bool = True,
    **kwargs
) -> ReductionResult:
    """
    Convenience function to reduce dimensions in one call.
    
    Args:
        X: Input embeddings (n_samples, n_features).
        method: Algorithm name (e.g., "umap", "tsne", "pca").
        n_components: Number of output dimensions.
        random_state: Random seed for reproducibility.
        verbose: Show progress for long-running algorithms.
        **kwargs: Algorithm-specific parameters.
        
    Returns:
        ReductionResult with reduced embeddings and metadata.
        
    Example:
        >>> result = reduce_dimensions(embeddings, method="pca", n_components=15)
        >>> print(result.shape)  # (n_items, 15)
    """
    reducer = get_reducer(
        method=method,
        n_components=n_components,
        random_state=random_state,
        verbose=verbose,
        **kwargs
    )
    
    reduced = reducer.fit_transform(X)
    
    return ReductionResult(
        embeddings=reduced,
        method=method,
        n_components=n_components,
        params=reducer.get_params(),
        metadata={"input_shape": X.shape, "output_shape": reduced.shape}
    )


def compare_methods(
    X: np.ndarray,
    methods: Optional[List[str]] = None,
    n_components: int = 2,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, ReductionResult]:
    """
    Run multiple DR methods on the same data for comparison.
    
    Args:
        X: Input embeddings.
        methods: List of methods to compare. If None, uses all core methods.
        n_components: Number of output dimensions.
        random_state: Random seed for reproducibility.
        verbose: Show progress.
        
    Returns:
        Dictionary mapping method name to ReductionResult.
    """
    if methods is None:
        methods = CORE_METHODS
    
    # Filter methods that don't support the requested n_components
    valid_methods = []
    for m in methods:
        reducer_class = REDUCER_REGISTRY.get(m)
        if reducer_class and (reducer_class.supports_high_dim or n_components <= 3):
            valid_methods.append(m)
        elif verbose:
            print(f"  Skipping {m}: doesn't support n_components={n_components}")
    
    results = {}
    
    iterator = tqdm(valid_methods, desc="Running DR methods") if verbose else valid_methods
    for method in iterator:
        try:
            result = reduce_dimensions(
                X,
                method=method,
                n_components=n_components,
                random_state=random_state,
                verbose=False  # Disable individual verbosity
            )
            results[method] = result
        except Exception as e:
            if verbose:
                print(f"  {method} failed: {e}")
    
    return results


# ============================================================================
# Legacy Compatibility
# ============================================================================

def umap_reduce(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_components: int = 15,
    metric: str = "cosine",
    random_state: int = 10,
) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    
    Prefer using get_reducer() or reduce_dimensions() for new code.
    """
    reducer = UMAPReducer(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    return reducer.fit_transform(X)
