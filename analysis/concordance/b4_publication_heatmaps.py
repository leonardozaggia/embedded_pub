
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Standalone publication-ready heatmap generator.

Generates three raw-count item-overlap heatmaps:
1) Theoretical vs Detected
2) Detected vs Empirical
3) Theoretical vs Empirical

Defaults are intentionally aligned with `b4_sankey.py`, so the script can be run
without any arguments.

Usage:
    python analysis/concordance/b4_publication_heatmaps.py
"""

import argparse
from pathlib import Path
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bayley_nlp"))
from utils import update_results_json, ensure_dir



CANONICAL_THEORETICAL_ORDER = [
    "working_memory",
    "flexibility_shift",
    "goal_directed_problem_solving",
    "attention",
    "higher_order_processing",
]

# Canonical display order for detected clusters — matches theoretical row order so
# the largest values lie as close to the diagonal as possible.  This is a
# visualization-only reordering; cluster composition is never changed.
CANONICAL_DETECTED_ORDER = ["WM", "FS", "GDPS", "ATT", "HOP"]

THEORETICAL_NAME_MAP = {
    "working memory": "WM",
    "flexibility shift": "FS",
    "goal directed problem solving": "GDPS",
    "attention": "ATT",
    "higher order processing": "HOP",
}

DETECTED_NAME_MAP = {
    "working memory": "WM",
    "w": "WM",
    "attention": "ATT",
    "a": "ATT",
    "higher order processing": "HOP",
    "higher order processes": "HOP",
    "h": "HOP",
    "flexibility shift": "FS",
    "f": "FS",
    "goal directed problem solving": "GDPS",
    "g": "GDPS",
}


def canonicalize_name(name: str, mapping: Dict[str, str]) -> str:
    """Map a label to canonical abbreviation when known."""
    key = str(name).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key in mapping:
        return mapping[key]
    return str(name)


def map_name_list(names: List[str], mapping: Dict[str, str]) -> List[str]:
    """Apply canonical mapping to a list of labels."""
    return [canonicalize_name(name, mapping) for name in names]


def resolve_paths(args, root: Path) -> Tuple[Path, Path, Path, Path]:
    """Resolve input/output paths with b4_sankey-compatible defaults."""
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "outputs" / "concordance" / "b4" / "publication_heatmaps"
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else root / "data" / "outputs" / "step2_b4_validation" / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
    )

    cluster_file = run_dir / "03_clustering" / "cluster_assignments.csv"
    cluster_labels_file = run_dir / "04_labeling" / "cluster_concept_labels.csv"
    reference_models = Path(args.reference_models) if args.reference_models else root / "config" / "reference_models_b4.yaml"

    return cluster_file, cluster_labels_file, reference_models, output_dir


def load_cluster_labels(cluster_labels_file: Path) -> Tuple[List[str], Dict[int, str]]:
    """Load detected cluster labels from step 04 outputs."""
    labels_path = Path(cluster_labels_file)

    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)

        if "cluster_id" in labels_df.columns and "top_concept" in labels_df.columns:
            label_dict = {int(cluster_id): str(concept) for cluster_id, concept in zip(labels_df["cluster_id"], labels_df["top_concept"])}
            label_list = [label_dict[idx] if label_dict[idx] else f"C{idx}" for idx in sorted(label_dict.keys())]
            return label_list, label_dict

        if "label" in labels_df.columns and "cluster" in labels_df.columns:
            label_dict = {int(cluster_id): str(label) for cluster_id, label in zip(labels_df["cluster"], labels_df["label"])}
            label_list = [label_dict[idx] for idx in sorted(label_dict.keys())]
            return label_list, label_dict

    raise FileNotFoundError(f"Could not parse cluster labels file: {cluster_labels_file}")


def load_reference_models(yaml_path: Path) -> Dict:
    """Load reference models YAML."""
    with open(yaml_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def reorder_theoretical_dict(theoretical_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Reorder theoretical factors to canonical order when present."""
    ordered = {}
    for key in CANONICAL_THEORETICAL_ORDER:
        if key in theoretical_dict:
            ordered[key] = theoretical_dict[key]
    for key, value in theoretical_dict.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _reorder_detected_to_canonical(
    detected_names: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Build a canonical display order for the detected clusters.

    Visual-only — does NOT change cluster composition or label assignments.
    Returns the name list in canonical order plus an old_to_new index array
    (old_to_new[old_cluster_id] = new_display_column) so that the col_labels /
    row_labels arrays can be remapped before passing to the heatmap renderer.
    """
    canonical = [n for n in CANONICAL_DETECTED_ORDER if n in detected_names]
    canonical += [n for n in detected_names if n not in CANONICAL_DETECTED_ORDER]
    old_to_new = np.array([canonical.index(name) for name in detected_names], dtype=int)
    return canonical, old_to_new


def _apply_permutation(labels: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    """Remap non-negative label indices through a permutation for display reordering."""
    result = labels.copy()
    mask = (labels >= 0) & (labels < len(old_to_new))
    result[mask] = old_to_new[labels[mask]]
    return result


def convert_factor_dict_to_item_factor_map(
    factor_dict: Dict[str, List[str]],
    all_items: List[str],
) -> Dict[str, List[int]]:
    """Map each item to one or more factor indices."""
    item_to_factors = {item_id: [] for item_id in all_items}
    for factor_idx, (_, items) in enumerate(factor_dict.items()):
        for item_id in items:
            if item_id in item_to_factors:
                item_to_factors[item_id].append(factor_idx)
    return item_to_factors


def expand_labels_for_multi_factor(
    all_items: List[str],
    detected_labels: np.ndarray,
    item_to_factors: Dict[str, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand (detected, reference) labels so multi-factor items appear once per factor."""
    expanded_detected, expanded_reference = [], []

    for item_idx, item_id in enumerate(all_items):
        factors = item_to_factors.get(item_id, [])

        if not factors:
            expanded_detected.append(int(detected_labels[item_idx]))
            expanded_reference.append(-1)
            continue

        for factor_idx in factors:
            expanded_detected.append(int(detected_labels[item_idx]))
            expanded_reference.append(int(factor_idx))

    return np.array(expanded_detected, dtype=int), np.array(expanded_reference, dtype=int)


def expand_pair_labels_for_multi_factor(
    all_items: List[str],
    left_item_to_factors: Dict[str, List[int]],
    right_item_to_factors: Dict[str, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand (left, right) labels for pairwise model comparisons with multi-factor items."""
    expanded_left, expanded_right = [], []

    for item_id in all_items:
        left_factors = left_item_to_factors.get(item_id, [])
        right_factors = right_item_to_factors.get(item_id, [])
        if not left_factors or not right_factors:
            continue

        for left_factor in left_factors:
            for right_factor in right_factors:
                expanded_left.append(int(left_factor))
                expanded_right.append(int(right_factor))

    return np.array(expanded_left, dtype=int), np.array(expanded_right, dtype=int)


def _compute_contingency_matrix(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Compute contingency matrix with fixed row/column cardinalities."""
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for row_label, col_label in zip(row_labels, col_labels):
        row_idx = int(row_label)
        col_idx = int(col_label)
        if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
            matrix[row_idx, col_idx] += 1
    return matrix


def _save_png_and_pdf(fig: plt.Figure, output_png_path: Path) -> None:
    """Save figure as publication PNG and vector PDF."""
    fig.savefig(output_png_path, dpi=600)
    fig.savefig(output_png_path.with_suffix(".pdf"))


def _format_axis_labels(labels: List[str], wrap_width: int = 20) -> List[str]:
    """Make labels publication-friendly for readability."""
    formatted = []
    for label in labels:
        clean = str(label).replace("_", " ").replace("-", " ").strip()
        formatted.append(textwrap.fill(clean, width=wrap_width))
    return formatted


def _compute_heatmap_metrics(matrix: np.ndarray, row_names: List[str], col_names: List[str]) -> dict:
    n_rows, n_cols = matrix.shape
    diag_vals = [float(matrix[i, i]) for i in range(min(n_rows, n_cols))]
    off_diag_vals = [float(matrix[i, j]) for i in range(n_rows) for j in range(n_cols) if i != j]
    within_mean = float(np.mean(diag_vals)) if diag_vals else 0.0
    between_mean = float(np.mean(off_diag_vals)) if off_diag_vals else 0.0
    separation_ratio = round(within_mean / between_mean, 3) if between_mean > 0 else None
    silhouette_per_factor: dict = {}
    for i, r_name in enumerate(row_names):
        match_j = next((j for j, c in enumerate(col_names) if c == r_name), None)
        if match_j is None:
            continue
        diag = float(matrix[i, match_j])
        offs = [float(matrix[i, j]) for j in range(n_cols) if j != match_j]
        max_off = max(offs) if offs else 0.0
        denom = max(diag, max_off)
        silhouette_per_factor[r_name] = round((diag - max_off) / denom, 4) if denom > 0 else 0.0
    mean_silhouette = round(float(np.mean(list(silhouette_per_factor.values()))), 4) if silhouette_per_factor else None
    top_off = sorted(
        [{"row": row_names[i] if i < len(row_names) else f"R{i}",
          "col": col_names[j] if j < len(col_names) else f"C{j}",
          "count": int(matrix[i, j])}
         for i in range(n_rows) for j in range(n_cols) if i != j and matrix[i, j] > 0],
        key=lambda x: -x["count"],
    )[:5]
    return {
        "within_cluster_mean": round(within_mean, 3),
        "between_cluster_mean": round(between_mean, 3),
        "separation_ratio": separation_ratio,
        "mean_silhouette": mean_silhouette,
        "per_factor_silhouette": silhouette_per_factor,
        "top_off_diagonal_pairs": top_off,
    }


def create_publication_raw_count_heatmap(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    row_axis_label: str,
    col_axis_label: str,
    output_png_path: Path,
) -> None:
    """Create publication-ready raw-count item-overlap heatmap."""
    matrix = _compute_contingency_matrix(
        row_labels=row_labels,
        col_labels=col_labels,
        n_rows=len(row_names),
        n_cols=len(col_names),
    )

    row_names_display = _format_axis_labels(row_names)
    col_names_display = _format_axis_labels(col_names, wrap_width=14)

    sns.set_theme(context="paper", style="white", font_scale=2.3)

    fig, ax_raw = plt.subplots(1, 1, figsize=(11.8, 8.6), constrained_layout=False)
    fig.subplots_adjust(left=0.26, right=0.91, bottom=0.16, top=0.95)

    sns.heatmap(
        matrix,
        ax=ax_raw,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.8,
        linecolor="white",
        xticklabels=col_names_display,
        yticklabels=row_names_display,
        cbar_kws={"label": "Item Count"},
        annot_kws={"fontsize": 21, "fontweight": "semibold"},
    )
    ax_raw.set_xlabel(col_axis_label, fontsize=25, labelpad=10)
    ax_raw.set_ylabel(row_axis_label, fontsize=25, labelpad=10)
    ax_raw.tick_params(axis="x", labelrotation=0, labelsize=20)
    ax_raw.tick_params(axis="y", labelrotation=0, labelsize=20)

    for spine in ax_raw.spines.values():
        spine.set_visible(False)

    _save_png_and_pdf(fig, output_png_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready overlap heatmaps")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional override for experiment run directory")
    parser.add_argument("--reference-models", type=str, default=None, help="Optional override for reference YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for output directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cluster_file, cluster_labels_file, reference_models_file, output_dir = resolve_paths(args, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster assignments file not found: {cluster_file}")
    if not cluster_labels_file.exists():
        raise FileNotFoundError(f"Cluster labels file not found: {cluster_labels_file}")
    if not reference_models_file.exists():
        raise FileNotFoundError(f"Reference models file not found: {reference_models_file}")

    detected_df = pd.read_csv(cluster_file)
    all_items = detected_df["item_id"].tolist()
    detected_labels = detected_df["cluster"].to_numpy(dtype=int)

    detected_names, _ = load_cluster_labels(cluster_labels_file)
    reference_models = load_reference_models(reference_models_file)

    theoretical_dict = {k: v for k, v in reference_models["theoretical"].items() if isinstance(v, list)}
    empirical_dict = {k: v for k, v in reference_models["empirical"].items() if isinstance(v, list)}
    theoretical_dict = reorder_theoretical_dict(theoretical_dict)

    theoretical_names = list(theoretical_dict.keys())
    empirical_names = list(empirical_dict.keys())
    theoretical_names = map_name_list(theoretical_names, THEORETICAL_NAME_MAP)
    detected_names = map_name_list(detected_names, DETECTED_NAME_MAP)

    # Build canonical display order for detected clusters (visual-only; no data changes).
    detected_names_display, det_old_to_new = _reorder_detected_to_canonical(detected_names)

    theoretical_map = convert_factor_dict_to_item_factor_map(theoretical_dict, all_items)
    empirical_map = convert_factor_dict_to_item_factor_map(empirical_dict, all_items)

    detected_expanded_theoretical, theoretical_expanded = expand_labels_for_multi_factor(all_items, detected_labels, theoretical_map)
    valid_theoretical = (theoretical_expanded >= 0) & (detected_expanded_theoretical >= 0)
    create_publication_raw_count_heatmap(
        row_labels=theoretical_expanded[valid_theoretical],
        col_labels=_apply_permutation(detected_expanded_theoretical[valid_theoretical], det_old_to_new),
        row_names=theoretical_names,
        col_names=detected_names_display,
        row_axis_label="Theoretical Factors",
        col_axis_label="Detected Clusters",
        output_png_path=output_dir / "confusion_matrix_theoretical_publication.png",
    )

    detected_expanded_empirical, empirical_expanded = expand_labels_for_multi_factor(all_items, detected_labels, empirical_map)
    valid_empirical = (empirical_expanded >= 0) & (detected_expanded_empirical >= 0)
    create_publication_raw_count_heatmap(
        row_labels=_apply_permutation(detected_expanded_empirical[valid_empirical], det_old_to_new),
        col_labels=empirical_expanded[valid_empirical],
        row_names=detected_names_display,
        col_names=empirical_names,
        row_axis_label="Detected Clusters",
        col_axis_label="Empirical Factors",
        output_png_path=output_dir / "confusion_matrix_detected_vs_empirical_publication.png",
    )

    theoretical_pair, empirical_pair = expand_pair_labels_for_multi_factor(all_items, theoretical_map, empirical_map)
    valid_pair = (theoretical_pair >= 0) & (empirical_pair >= 0)
    create_publication_raw_count_heatmap(
        row_labels=theoretical_pair[valid_pair],
        col_labels=empirical_pair[valid_pair],
        row_names=theoretical_names,
        col_names=empirical_names,
        row_axis_label="Theoretical Factors",
        col_axis_label="Empirical Factors",
        output_png_path=output_dir / "confusion_matrix_theoretical_vs_empirical_publication.png",
    )

    print("Publication heatmaps generated:")
    print(f"  - {output_dir / 'confusion_matrix_theoretical_publication.png'}")
    print(f"  - {output_dir / 'confusion_matrix_detected_vs_empirical_publication.png'}")
    print(f"  - {output_dir / 'confusion_matrix_theoretical_vs_empirical_publication.png'}")
    print(f"Output directory: {output_dir}")

    # ── Export structured metrics ────────────────────────────────────────────
    results_dir = root / "results" / "metrics"
    ensure_dir(str(results_dir))
    metrics_file = str(results_dir / "manuscript_metrics.json")

    mat_td = _compute_contingency_matrix(
        theoretical_expanded[valid_theoretical], detected_expanded_theoretical[valid_theoretical],
        len(theoretical_names), len(detected_names))
    update_results_json("Bayley4_Heatmap_Theo_vs_Det",
        _compute_heatmap_metrics(mat_td, theoretical_names, detected_names),
        filename=metrics_file)

    mat_de = _compute_contingency_matrix(
        detected_expanded_empirical[valid_empirical], empirical_expanded[valid_empirical],
        len(detected_names), len(empirical_names))
    update_results_json("Bayley4_Heatmap_Det_vs_Emp",
        _compute_heatmap_metrics(mat_de, detected_names, empirical_names),
        filename=metrics_file)

    mat_te = _compute_contingency_matrix(
        theoretical_pair[valid_pair], empirical_pair[valid_pair],
        len(theoretical_names), len(empirical_names))
    update_results_json("Bayley4_Heatmap_Theo_vs_Emp",
        _compute_heatmap_metrics(mat_te, theoretical_names, empirical_names),
        filename=metrics_file)


if __name__ == "__main__":
    main()
