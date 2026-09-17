from typing import Optional, Tuple, List, Dict
import math
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.spatial.distance import cdist


def parallel_analysis(
    S: np.ndarray,
    n_iter: int = 1000,
    random_state: int = 42,
    percentile: float = 95.0,
    p_eff: int = None,
    simulate: str = "corr",
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Horn's parallel analysis on a correlation/similarity matrix with percentile cutoff.

    Args:
        S: square similarity/correlation matrix (items x items).
        n_iter: number of null draws.
        random_state: RNG seed.
        percentile: cutoff in the null eigenvalue distribution (e.g., 95).
        p_eff: effective sample size for the null. For embeddings, pass the embedding
               dimensionality (e.g., 384/512/768). If None, a conservative default is used.
        simulate: 'corr' (simulate iid normals -> Pearson R) or
                  'cosine' (simulate random embeddings -> cosine matrix).

    Returns:
        (n_factors_suggested, eig_obs_desc, eig_cut_desc)
    """
    rng = np.random.default_rng(random_state)
    n = S.shape[0]
    eig_obs = np.linalg.eigvalsh(S)[::-1]

    if p_eff is None:
        p_eff = min(2048, max(100, 5 * n))  # conservative fallback

    eig_null = np.empty((n_iter, n))
    for b in range(n_iter):
        if simulate == "corr":
            # p_eff observations of n uncorrelated variables -> Pearson R
            X = rng.standard_normal(size=(p_eff, n))
            X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
            Rb = np.corrcoef(X, rowvar=False)
        elif simulate == "cosine":
            # n random d-dimensional embeddings -> cosine similarity
            d = int(p_eff)
            Z = rng.standard_normal(size=(n, d))
            Z = Z - Z.mean(axis=0, keepdims=True)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)
            Rb = Z @ Z.T
            np.fill_diagonal(Rb, 1.0)
        else:
            raise ValueError("simulate must be 'corr' or 'cosine'.")
        eig_null[b] = np.linalg.eigvalsh(Rb)[::-1]

    eig_cut = np.percentile(eig_null, percentile, axis=0)
    n_f = int(np.sum(eig_obs > eig_cut))
    return n_f, eig_obs, eig_cut


def _varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = svd(
            Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vh
        d = s.sum()
        if d_old and (d - d_old) < tol:
            break
    return Phi @ R, R


def pca_with_rotation(S: np.ndarray, n_factors: int, rotation: str = "varimax") -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """PCA on S with rotation.

    Args:
        S: Square similarity/correlation matrix.
        n_factors: Number of components to retain.
        rotation: 'varimax', 'promax', or 'none'.

    Returns:
        (L_rot, Rmat, evals_desc, evecs_desc)
    """
    evals, evecs = np.linalg.eigh(S)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    L_unrot = evecs[:, :n_factors] * np.sqrt(evals[:n_factors])

    rotation = (rotation or "varimax").lower()
    if rotation == "varimax":
        L_rot, Rmat = _varimax(L_unrot.copy())
    elif rotation == "promax":
        from factor_analyzer import Rotator

        L_rot = Rotator(method="promax").fit_transform(L_unrot)
        Rmat = None
    else:
        L_rot, Rmat = L_unrot.copy(), None
    return L_rot, Rmat, evals, evecs


def assign_primary(loadings: np.ndarray) -> np.ndarray:
    """Primary factor index (argmax by absolute loading) per item."""
    return np.argmax(np.abs(loadings), axis=1)


def mcdonald_omega_from_loadings(L: np.ndarray, items_idx: np.ndarray) -> float:
    """Approximate McDonald's omega for a set of items given factor loadings.

    Uses the column with the largest mean absolute loading among selected items.
    """
    if len(items_idx) < 3:
        return math.nan
    Lk = L[items_idx, :]
    col = int(np.argmax(np.mean(np.abs(Lk), axis=0)))
    lam = Lk[:, col]
    h2 = lam ** 2
    u = 1 - h2
    num = (lam.sum()) ** 2
    den = num + u.sum()
    return float(num / den)


# ----------------------------
# Dimensionality selection
# ----------------------------

def broken_stick_thresholds(n: int) -> np.ndarray:
    """Broken-stick expected eigenvalues for ranks 1..n (sum to 1)."""
    ranks = np.arange(1, n + 1)
    bs = np.array([np.sum(1.0 / np.arange(k, n + 1)) for k in ranks]) / n
    return bs


def broken_stick_k(eigenvalues: np.ndarray) -> int:
    """Suggest k where observed eigenvalues exceed broken-stick expectations.

    Assumes eigenvalues are descending and scaled such that sum equals n (as for
    a correlation-like matrix). We compare normalized eigenvalues to BS thresholds.
    """
    ev = np.asarray(eigenvalues).astype(float)
    n = ev.size
    if n == 0:
        return 1
    # Normalize to sum to n to mimic correlation spectrum scale
    if ev.sum() > 0:
        ev_norm = ev / ev.sum() * n
    else:
        ev_norm = ev
    bs = broken_stick_thresholds(n)
    k = int(np.sum(ev_norm / n > bs))
    return max(1, k)


def elbow_k(eigenvalues: np.ndarray) -> int:
    """Knee/elbow via maximum distance to the line between first and last point.

    Returns index (1-based) of the elbow on the scree curve.
    """
    ev = np.asarray(eigenvalues).astype(float)
    n = ev.size
    if n <= 2:
        return max(1, n)
    # Points
    x = np.arange(1, n + 1)
    y = ev
    # Line from first to last
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    # Distance from each point to the line
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    d = num / (den if den != 0 else 1.0)
    k = int(np.argmax(d) + 1)
    return max(1, k)


def ppca_bic_k(X: np.ndarray, k_min: int = 1, k_max: Optional[int] = None) -> int:
    """Select k by PPCA BIC on features X (items x features).

    Simple PPCA approximation: fit PCA, estimate noise variance from trailing
    eigenvalues, compute log-likelihood and BIC per k; pick k with min BIC.
    """
    from numpy.linalg import svd as _svd

    n, d = X.shape
    if k_max is None:
        k_max = max(k_min, min(min(n, d) - 1, 10))
    # Center features
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD for eigenvalues
    U, Svals, Vt = _svd(Xc, full_matrices=False)
    eig = (Svals ** 2) / (n - 1)
    ks = list(range(int(k_min), int(k_max) + 1))
    best_k = ks[0]
    best_bic = float("inf")

    for k in ks:
        k = int(k)
        if k >= min(n, d):
            continue
        # Noise variance estimate: mean of trailing eigenvalues
        if k < len(eig):
            sigma2 = float(np.mean(eig[k:]))
        else:
            sigma2 = 1e-8
        sigma2 = max(sigma2, 1e-8)
        # Log-likelihood under PPCA (Tipping & Bishop):
        # log L = -n/2 [ d*log(2pi) + sum_i log(lambda_i) + d + ... ],
        # Using approximation with eigenvalues: see Bishop PRML ch. 12.
        # We use: logL = -n/2 [ d*log(2pi) + (d - k)*log(sigma2) + sum_{i=1..k} log(alpha_i) + d ]
        # where alpha_i = eigen_i for top k.
        top = eig[:k] if k > 0 else np.array([])
        term = (d - k) * math.log(sigma2) + np.sum(np.log(np.maximum(top, 1e-12))) + d
        logL = -0.5 * n * (d * math.log(2 * math.pi) + term)
        # Number of parameters in PPCA: W (d*k) minus rotational df (k*(k-1)/2) + mean (d) + sigma2 (1)
        p = d * k - k * (k - 1) // 2 + d + 1
        bic = -2.0 * logL + p * math.log(n)
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return int(best_k)


def rv_coefficient(A: np.ndarray, B: np.ndarray) -> float:
    """RV coefficient between two square similarity/covariance matrices.
    Ranges [0,1]; 1 indicates identical up to scaling.
    """
    A0 = A - A.mean()
    B0 = B - B.mean()
    num = np.sum(A0 * B0)
    den = math.sqrt(np.sum(A0 * A0) * np.sum(B0 * B0))
    if den <= 0:
        return float("nan")
    return float(num / den)


def run_kmeans(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    k_min: int = 2,
    k_max: Optional[int] = None,
) -> np.ndarray:
    """KMeans clustering on features X.

    If ``n_clusters`` is None, choose k via silhouette score over k in [k_min, k_max].
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    n_samples = int(X.shape[0])
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    if n_clusters is None:
        # Define a reasonable k search range
        if k_max is None:
            # cap by number of samples and keep search modest
            k_max = max(k_min, min(10, n_samples - 1))
        k_candidates = [k for k in range(int(k_min), int(k_max) + 1) if k >= 2 and k < n_samples]
        best_k = None
        best_score = -np.inf
        for k in k_candidates:
            try:
                try:
                    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
                except TypeError:
                    km = KMeans(n_clusters=k, random_state=random_state)
                labels = km.fit_predict(X)
                # Silhouette with euclidean on feature space
                score = silhouette_score(X, labels, metric="euclidean")
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        # Fallback if no score computed
        n_clusters = best_k if best_k is not None else max(2, min(3, n_samples))

    try:
        km_final = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
    except TypeError:
        km_final = KMeans(n_clusters=int(n_clusters), random_state=random_state)
    return km_final.fit_predict(X)


def run_spectral(
    S: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    k_min: int = 2,
    k_max: Optional[int] = None,
    method: str = "eigengap",
) -> np.ndarray:
    """Spectral clustering on a precomputed similarity matrix S.

    If ``n_clusters`` is None, estimate k using eigengap on the normalized
    graph Laplacian (fallback to silhouette over [k_min, k_max]).
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    n = int(S.shape[0])
    if n < 2:
        return np.zeros(n, dtype=int)

    # Ensure symmetric and non-negative affinity
    A = 0.5 * (S + S.T)
    A = np.where(A < 0, 0.0, A)
    np.fill_diagonal(A, 1.0)

    def _eigengap_k(Aff: np.ndarray, kmin: int, kmax: int) -> int:
        # Normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
        d = np.sum(Aff, axis=1)
        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(d)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(Aff.shape[0]) - (D_inv_sqrt @ Aff @ D_inv_sqrt)
        # Eigenvalues ascending
        evals = np.linalg.eigvalsh(L)
        evals = np.sort(np.real(evals))
        # Gaps between consecutive eigenvalues
        gaps = np.diff(evals)
        # Consider gaps at indices corresponding to k-1
        kmin = max(2, int(kmin))
        kmax = min(int(kmax), Aff.shape[0] - 1)
        if kmax < kmin:
            kmax = kmin
        best_k = kmin
        best_gap = -np.inf
        for k in range(kmin, kmax + 1):
            idx = k - 1
            if 0 <= idx < len(gaps):
                g = gaps[idx]
                if np.isfinite(g) and g > best_gap:
                    best_gap = g
                    best_k = k
        return int(best_k)

    if n_clusters is None:
        if k_max is None:
            k_max = max(k_min, min(10, n - 1))
        # Try eigengap first
        try:
            k_eig = _eigengap_k(A, k_min, k_max)
            n_clusters = int(k_eig)
        except Exception:
            n_clusters = None

        # Fallback: silhouette search over k on precomputed distances
        if n_clusters is None or n_clusters < 2:
            print("Fall-back, spectral clustering failed")
            k_candidates = [k for k in range(int(k_min), int(k_max) + 1) if k >= 2 and k < n]
            best_k = None
            best_score = -np.inf
            # Distance matrix from similarity
            Dmat = 1.0 - (A / (A.max() if A.max() > 0 else 1.0))
            for k in k_candidates:
                try:
                    spec = SpectralClustering(
                        n_clusters=k,
                        affinity="precomputed",
                        random_state=random_state,
                        assign_labels="kmeans",
                    )
                    labels = spec.fit_predict(A)
                    score = silhouette_score(Dmat, labels, metric="precomputed")
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_k = k
                except Exception:
                    continue
            n_clusters = best_k if best_k is not None else max(2, min(3, n))

    spec_final = SpectralClustering(
        n_clusters=int(n_clusters),
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    return spec_final.fit_predict(A)


def human_factor_loadings(
    excel_path: str,
    sheet_name: str,
    n_factors: int,
    rotation: str = "varimax",
    item_columns: Optional[List[str]] = None,
    auto_from: int = 34,
    drop_last: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute rotated loadings from human response Excel sheet.
    Returns (loadings_df, columns_used).
    """
    import re as _re
    import pandas as _pd

    df_h = _pd.read_excel(excel_path, sheet_name=sheet_name).iloc[1:].copy()

    if item_columns is None:
        cols = df_h.columns.astype(str).tolist()
        cognitive_items = [c for c in cols if _re.match(r"^bsid_cog", c)]
        exclude_c = {c for c in cols if _re.match(r"^bsid_cog_", c)}
        cognitive_items = [c for c in cognitive_items if c not in exclude_c]
        start_idx = max(1, int(auto_from))
        start_idx = min(start_idx, len(cognitive_items))
        item_columns = cognitive_items[start_idx - 1 :]

        for c in item_columns:
            df_h[c] = _pd.to_numeric(df_h[c], errors="coerce")
        item_columns = [c for c in item_columns if df_h[c].nunique(dropna=True) > 1]
        if len(item_columns) > drop_last:
            item_columns = item_columns[: len(item_columns) - drop_last]

    cols_use = [c for c in item_columns if c in df_h.columns]
    if len(cols_use) == 0:
        raise ValueError("No matching HUMAN_ITEM_COLUMNS found in Excel.")

    X = df_h[cols_use].apply(_pd.to_numeric, errors="coerce")
    X = X.loc[~X.isna().all(axis=1)]
    R = X.corr(method="pearson")

    ev_h, evc_h = np.linalg.eigh(R.values)
    idxh = np.argsort(ev_h)[::-1]
    ev_h, evc_h = ev_h[idxh], evc_h[:, idxh]
    Lh_unrot = evc_h[:, :n_factors] * np.sqrt(ev_h[:n_factors])

    rotation = (rotation or "varimax").lower()
    if rotation == "varimax":
        Lh_rot, _ = _varimax(Lh_unrot.copy())
    elif rotation == "promax":
        from factor_analyzer import Rotator

        Lh_rot = Rotator(method="promax").fit_transform(Lh_unrot)
    else:
        Lh_rot = Lh_unrot.copy()

    loadings_h = pd.DataFrame(Lh_rot, index=R.index, columns=[f"PC{i+1}" for i in range(n_factors)])
    return loadings_h, cols_use


def congruence_matrix(emb_loadings: pd.DataFrame, human_loadings: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations of absolute loadings between embedding and human PCs."""
    if emb_loadings.shape[0] != human_loadings.shape[0]:
        raise ValueError("Different item counts; cannot compute congruence.")
    from scipy.stats import spearmanr

    n = emb_loadings.shape[1]
    m = human_loadings.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            rho, _ = spearmanr(np.abs(emb_loadings.iloc[:, i]), np.abs(human_loadings.iloc[:, j]))
            C[i, j] = rho
    return pd.DataFrame(
        C,
        index=[f"Emb_PC{i+1}" for i in range(n)],
        columns=[f"Hum_PC{j+1}" for j in range(m)],
    )

def _init_from_eigendecomp(R: np.ndarray, k: int) -> np.ndarray:
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1][:k]
    w = np.clip(w[idx], 1e-8, None)
    v = v[:, idx]
    return v * np.sqrt(w)

def _fit_lowrank_corr(R: np.ndarray, k: int, mask_train: np.ndarray, n_iter: int = 800, lr: float = 0.05, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    m = R.shape[0]
    U = _init_from_eigendecomp(R, k)
    M = mask_train.astype(float).copy()
    np.fill_diagonal(M, 0.0)
    for _ in range(n_iter):
        RU = U @ U.T
        G = 2.0 * ((RU - R) * M) @ U
        U -= lr * G
        U *= (1.0 / max(1.0, np.linalg.norm(U) / (m ** 0.5)))
    return U

def _masked_mse(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    D = (A - B)[mask.astype(bool)]
    return float(np.mean(D**2)) if D.size else np.nan

def cv_rank_from_corr(R: np.ndarray, k_max: int = 20, holdout_frac: float = 0.15, n_splits: int = 5,
                      n_iter: int = 800, lr: float = 0.05, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    m = R.shape[0]
    k_max = int(min(k_max, m - 1))
    iu, ju = np.triu_indices(m, k=1)
    pairs = np.stack([iu, ju], axis=1)
    n_pairs = pairs.shape[0]
    errs = {k: [] for k in range(1, k_max + 1)}
    for _ in range(n_splits):
        idx = rng.choice(n_pairs, size=int(holdout_frac * n_pairs), replace=False)
        mask_train = np.ones((m, m), dtype=bool)
        for (i, j) in pairs[idx]:
            mask_train[i, j] = mask_train[j, i] = False
        for k in range(1, k_max + 1):
            U = _fit_lowrank_corr(R, k, mask_train, n_iter=n_iter, lr=lr,
                                  random_state=rng.integers(1 << 30))
            R_hat = U @ U.T
            mask_val = (~mask_train).copy()
            np.fill_diagonal(mask_val, False)
            errs[k].append(_masked_mse(R, R_hat, mask_val))
    ks = np.array(sorted(errs.keys()))
    means = np.array([np.nanmean(errs[k]) for k in ks])
    ses = np.array([np.nanstd(errs[k], ddof=1) / np.sqrt(n_splits) for k in ks])
    k_min = ks[np.nanargmin(means)]
    threshold = means[ks == k_min][0] + ses[ks == k_min][0]  # one-SE rule
    k_best = int(ks[np.where(means <= threshold)[0][0]])
    stats = {"k": ks, "cv_mean_mse": means, "cv_se": ses, "k_min_err": int(k_min), "k_one_se": int(k_best)}
    return k_best, errs, stats

def _tucker_congruence(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return (num / den) if den > 0 else 0.0

def bootstrap_stability(S: np.ndarray, k: int, n_reps: int = 100, rotate: bool = True, random_state: int = 42, phi_threshold: float = 0.90):
    rng = np.random.default_rng(random_state)
    m = S.shape[0]
    # base loadings on full set
    L_base, _, _, _ = pca_with_rotation(S, n_factors=k, rotation="varimax" if rotate else "none")
    congruences = []
    for _ in range(n_reps):
        idx = rng.integers(0, m, size=m)  # bootstrap items
        Sb = S[np.ix_(idx, idx)]
        Lb, _, _, _ = pca_with_rotation(Sb, n_factors=k, rotation="varimax" if rotate else "none")
        # rows that appear once -> map back to base space
        unique, counts = np.unique(idx, return_counts=True)
        mask_once = counts == 1
        rows = unique[mask_once]
        if rows.size < max(10, k):
            continue
        pos = {j: np.where(idx == j)[0][0] for j in rows}
        Lb_sub = Lb[[pos[j] for j in rows], :]
        L0_sub = L_base[rows, :]
        # greedy alignment by absolute congruence
        used = set()
        phis_r = []
        for i in range(k):
            best_j, best_phi = None, -1.0
            for j in range(k):
                if j in used:
                    continue
                phi = abs(_tucker_congruence(L0_sub[:, i], Lb_sub[:, j]))
                if phi > best_phi:
                    best_phi, best_j = phi, j
            used.add(best_j)
            phis_r.append(best_phi)
        congruences.append(phis_r)
    if not congruences:
        return {"median_phi": float("nan"), "per_comp_median": None, "stable": False}
    C = np.array(congruences)  # reps x k
    per_comp_median = np.nanmedian(C, axis=0)
    median_phi = float(np.nanmedian(per_comp_median))
    stable = bool(np.all(per_comp_median >= phi_threshold))
    return {"median_phi": median_phi, "per_comp_median": per_comp_median, "stable": stable}

def select_k_non_circular(R: np.ndarray, embed_dim: int = None,
                          pa_iter: int = 1000, cv_kmax: int = 20, seed: int = 42):
    # 1) Parallel analysis with 95th percentile
    k_pa, eig_obs, eig_cut = parallel_analysis(
        R, n_iter=pa_iter, random_state=seed, percentile=95.0,
        p_eff=embed_dim, simulate="corr"
    )

    # 2) Cross-validated reconstruction (one-SE)
    k_cv, errs, stats = cv_rank_from_corr(
        R, k_max=min(cv_kmax, R.shape[0] - 1),
        holdout_frac=0.15, n_splits=6,
        n_iter=600, lr=0.05, random_state=seed
    )

    # 3) Candidate = rounded median of PA and CV
    if k_pa > 0:
        k_candidate = int(round(np.median([k_pa, k_cv])))
    else:
        k_candidate = int(k_cv)
    k_candidate = max(1, min(k_candidate, R.shape[0] - 1))

    # 4) Stability check: choose the LARGEST k ≤ candidate that is stable
    k_stable = None
    for k in range(k_candidate, 0, -1):
        stab = bootstrap_stability(R, k=k, n_reps=150, rotate=True,
                                   random_state=seed, phi_threshold=0.90)
        if stab["stable"]:
            k_stable = k
            stab_final = stab
            break

    if k_stable is None:
        k_stable, stab_final = 1, {"median_phi": float("nan"),
                                   "per_comp_median": None,
                                   "stable": False}

    return {
        "k": int(k_stable),                # largest stable k
        "k_candidate": int(k_candidate),   # raw candidate before stability
        "k_pa": int(k_pa),
        "k_cv_one_se": int(k_cv),
        "cv_stats": stats,
        "eig_obs": eig_obs,
        "eig_cut": eig_cut,
        "stability": {
            "median_phi": float(stab_final["median_phi"]),
            "per_comp_median": None if stab_final["per_comp_median"] is None
                               else [float(x) for x in stab_final["per_comp_median"]],
            "stable": bool(stab_final["stable"]),
        },
    }

