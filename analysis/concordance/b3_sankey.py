
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Agreement-Weighted Sankey Visualization — B3 Edition.
Uses step3_b3_bottomup (detected) vs step1_translation (theoretical).
Single comparison: Theoretical → Detected.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import re

import plotly.graph_objects as go
import sys
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bayley_nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import update_results_json, ensure_dir


# --- CONFIGURATION & ORDERING ---
CANONICAL_THEORETICAL_ORDER = [
    "working_memory",
    "flexibility_shift",
    "goal_directed_problem_solving",
    "attention",
    "higher_order_processing",
]

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


def load_theoretical_from_b3_csv(assignments_csv: Path) -> Dict[str, List[str]]:
    """Load the translated (Objective 1) factor assignments for all 91 items.

    Reads results/tables/bayley3_item_domain_assignments.csv (cascade pairing +
    centroid assignment + expert review) via analysis/concordance/_translated.py.
    """
    from _translated import load_theoretical_dict
    return load_theoretical_dict(assignments_csv.parents[2])


def canonicalize_name(name: str, mapping: Dict[str, str]) -> str:
    key = str(name).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key in mapping:
        return mapping[key]
    return str(name)


def map_name_list(names: List[str], mapping: Dict[str, str]) -> List[str]:
    return [canonicalize_name(name, mapping) for name in names]


def reorder_theoretical_dict(theoretical_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    ordered = {}
    for key in CANONICAL_THEORETICAL_ORDER:
        if key in theoretical_dict:
            ordered[key] = theoretical_dict[key]
    for key, value in theoretical_dict.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def load_cluster_labels(cluster_labels_file: Path) -> Tuple[List[str], Dict[int, str]]:
    labels_path = Path(cluster_labels_file)
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if "cluster_id" in labels_df.columns and "top_concept" in labels_df.columns:
            label_dict = {int(k): str(v) for k, v in zip(labels_df["cluster_id"], labels_df["top_concept"])}
            return [label_dict[i][0].upper() if label_dict[i] else f"C{i}" for i in sorted(label_dict.keys())], label_dict
        if "label" in labels_df.columns and "cluster" in labels_df.columns:
            label_dict = {int(k): str(v) for k, v in zip(labels_df["cluster"], labels_df["label"])}
            return [label_dict[i] for i in sorted(label_dict.keys())], label_dict
    raise FileNotFoundError("Could not parse cluster labels file.")


def abbreviate_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", str(label).lower().replace("_", " ").replace("-", " ")).strip()
    overrides = {
        "attention": "ATT",
        "working memory": "WM",
        "goal directed problem solving": "GDPS",
        "flexibility shift": "FS",
        "higher order processing": "HOP",
        "higher order processes": "HOP",
    }
    if normalized in overrides:
        return overrides[normalized]
    if normalized in DETECTED_NAME_MAP:
        return DETECTED_NAME_MAP[normalized]
    if normalized in THEORETICAL_NAME_MAP:
        return THEORETICAL_NAME_MAP[normalized]
    if normalized.startswith("factor "):
        return normalized.replace("factor ", "F").upper()
    parts = [part for part in str(label).replace("-", "_").split("_") if part]
    if not parts:
        return str(label)
    if len(parts) == 1:
        return parts[0][:8].upper()
    return "".join(part[0].upper() for part in parts)


def build_detected_colors_from_theoretical(
    detected_names: List[str],
    theoretical_names: List[str],
    theoretical_colors: List[str],
    fallback_palette: List[str],
) -> List[str]:
    theoretical_color_by_abbrev = {
        abbreviate_label(name): color for name, color in zip(theoretical_names, theoretical_colors)
    }
    colors: List[str] = []
    fallback_index = 0
    for detected_name in detected_names:
        detected_abbrev = abbreviate_label(detected_name)
        if detected_abbrev in theoretical_color_by_abbrev:
            colors.append(theoretical_color_by_abbrev[detected_abbrev])
        else:
            colors.append(fallback_palette[fallback_index % len(fallback_palette)])
            fallback_index += 1
    return colors


def build_palette(n_colors: int) -> List[str]:
    base = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    return (base * ((n_colors // len(base)) + 1))[:n_colors]


def pastelize_hex(hex_color: str, amount: float = 0.58) -> str:
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return str(hex_color)
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r_p = int(round(r + (255 - r) * amount))
    g_p = int(round(g + (255 - g) * amount))
    b_p = int(round(b + (255 - b) * amount))
    return f"#{r_p:02X}{g_p:02X}{b_p:02X}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(180, 180, 180, {alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def convert_factor_dict_to_item_factor_map(factor_dict: Dict[str, List[str]], all_items: List[str]) -> Dict[str, List[int]]:
    item_to_factors = {item_id: [] for item_id in all_items}
    for factor_idx, (_, items) in enumerate(factor_dict.items()):
        for item_id in items:
            if item_id in item_to_factors:
                item_to_factors[item_id].append(factor_idx)
    return item_to_factors


def expand_labels_for_multi_factor(all_items: List[str], detected_labels: np.ndarray, item_to_factors: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_agreement_sankey_plotly(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_names: List[str],
    right_names: List[str],
    left_colors: List[str],
    right_colors: List[str],
    title: str,
    output_path: Path,
    fig_width=1020,
    fig_height=640,
    left_node_totals: np.ndarray = None,
    show_title: bool = False,
    use_agreement_dimming: bool = True,
) -> None:
    valid_mask = (left_labels >= 0) & (right_labels >= 0)
    ll_v, rl_v = left_labels[valid_mask], right_labels[valid_mask]

    n_left, n_right = len(left_names), len(right_names)
    counts = np.zeros((n_left, n_right), dtype=int)
    for l, r in zip(ll_v, rl_v):
        counts[int(l), int(r)] += 1

    connected_left = counts.sum(axis=1)

    if left_node_totals is None:
        left_totals = connected_left.astype(float)
    else:
        left_totals = np.maximum(left_node_totals.astype(float), connected_left)

    source, target, value, link_color = [], [], [], []

    for i in range(n_left):
        for j in range(n_right):
            if counts[i, j] > 0:
                source.append(i)
                target.append(j + n_left)
                value.append(counts[i, j])

                name_l_clean = left_names[i].split(" (n=")[0]
                name_r_clean = right_names[j].split(" (n=")[0]

                if not use_agreement_dimming:
                    link_color.append(hex_to_rgba(left_colors[i], 0.42))
                elif abbreviate_label(name_l_clean) == abbreviate_label(name_r_clean):
                    link_color.append(hex_to_rgba(left_colors[i], 0.42))
                else:
                    link_color.append(hex_to_rgba(left_colors[i], 0.10))

    node_labels = [f"{name} (n={int(left_totals[i])})" for i, name in enumerate(left_names)] + \
                  [f"{name} (n={int(counts.sum(axis=0)[j])})" for j, name in enumerate(right_names)]

    node_left_colors = [pastelize_hex(color, amount=0.62) for color in left_colors]
    node_right_colors = [pastelize_hex(color, amount=0.62) for color in right_colors]
    node_colors = node_left_colors + node_right_colors

    def get_y_pos(totals, gap=0.02):
        if len(totals) == 1:
            return [0.5]
        t_sum = totals.sum()
        usable = 0.9 - (gap * (len(totals) - 1))
        heights = (totals / t_sum) * usable
        pos, cursor = [], 0.05
        for h in heights:
            pos.append(cursor + h / 2)
            cursor += h + gap
        return pos

    node_x = [0.05] * n_left + [0.95] * n_right
    node_y = get_y_pos(left_totals) + get_y_pos(counts.sum(axis=0))

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(pad=25, thickness=20, label=node_labels, color=node_colors, x=node_x, y=node_y),
        link=dict(source=source, target=target, value=value, color=link_color)
    )])

    title_text = title if show_title else ""
    top_margin = 60 if show_title else 12
    fig.update_layout(
        title_text=title_text,
        font=dict(size=26, family="Arial"),
        width=fig_width,
        height=fig_height,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=top_margin, b=10)
    )
    fig.write_image(str(output_path), scale=2)


def resolve_paths(args, root):
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "outputs" / "concordance" / "b3"
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else root / "data" / "outputs" / "step3_b3_bottomup" / "all_mpnet_base_v2_kmeans_consensus_no_dr"
    )

    cluster_file = run_dir / "03_clustering" / "cluster_assignments.csv"
    cluster_labels_file = run_dir / "04_labeling" / "cluster_concept_labels.csv"
    assignments_csv = root / "results" / "tables" / "bayley3_item_domain_assignments.csv"

    return cluster_file, cluster_labels_file, assignments_csv, output_dir


def _sankey_metrics(left_labels, right_labels, left_names, right_names):
    n_l, n_r = len(left_names), len(right_names)
    mat = np.zeros((n_l, n_r), dtype=int)
    for lbl, rbl in zip(left_labels, right_labels):
        mat[int(lbl), int(rbl)] += 1
    ari = round(float(adjusted_rand_score(left_labels, right_labels)), 4)
    nmi = round(float(normalized_mutual_info_score(left_labels, right_labels)), 4)
    left_totals = {name: int(mat[i].sum()) for i, name in enumerate(left_names)}
    right_totals = {name: int(mat[:, j].sum()) for j, name in enumerate(right_names)}
    transitions, flags = [], []
    for i, l_name in enumerate(left_names):
        l_total = int(mat[i].sum())
        for j, r_name in enumerate(right_names):
            cnt = int(mat[i, j])
            if cnt > 0:
                pct = round(100 * cnt / l_total, 1) if l_total > 0 else 0.0
                transitions.append({
                    "from": l_name, "to": r_name,
                    "n": cnt, "pct_of_left": pct,
                    "is_match": l_name == r_name,
                })
                if l_name != r_name and pct >= 20.0:
                    flags.append(f"{pct}% of {l_name} → {r_name} (N={cnt})")
    return {
        "ARI": ari, "NMI": nmi,
        "left_marginal_totals": left_totals,
        "right_marginal_totals": right_totals,
        "transition_proportions": transitions,
        "major_reallocation_flags": flags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--show-titles", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cluster_f, label_f, assignments_csv, out_d = resolve_paths(args, root)
    out_d.mkdir(parents=True, exist_ok=True)

    # Load detected solution
    df = pd.read_csv(cluster_f)
    items, det_labels = df["item_id"].tolist(), df["cluster"].to_numpy()
    _, cluster_labels = load_cluster_labels(label_f)

    # Load theoretical solution from B3 CSV files
    theo_dict = load_theoretical_from_b3_csv(assignments_csv)
    theo_dict = reorder_theoretical_dict(theo_dict)

    # Colors
    theo_names = map_name_list(list(theo_dict.keys()), THEORETICAL_NAME_MAP)
    detected_names_raw = [cluster_labels.get(i, f"C{i}") for i in range(int(max(det_labels)) + 1)]
    detected_names = map_name_list(detected_names_raw, DETECTED_NAME_MAP)

    palette = build_palette(max(len(theo_names), len(detected_names)) + 8)
    theo_colors = palette[:len(theo_names)]
    det_colors = build_detected_colors_from_theoretical(
        detected_names=detected_names,
        theoretical_names=theo_names,
        theoretical_colors=theo_colors,
        fallback_palette=palette,
    )

    # Mappings
    theo_map = convert_factor_dict_to_item_factor_map(theo_dict, items)

    # PLOT: THEORETICAL → DETECTED
    d_exp, t_exp = expand_labels_for_multi_factor(items, det_labels, theo_map)
    v = t_exp >= 0
    plot_agreement_sankey_plotly(
        t_exp[v], d_exp[v],
        theo_names, detected_names,
        theo_colors, det_colors,
        "Theoretical → Detected (Weighted)",
        out_d / "sankey_theo_to_det.png",
        show_title=args.show_titles,
        use_agreement_dimming=True,
    )

    print(f"Sankey generated: {out_d / 'sankey_theo_to_det.png'}")

    # ── Export structured metrics ────────────────────────────────────────────
    results_dir = root / "results" / "metrics"
    ensure_dir(str(results_dir))
    metrics_file = str(results_dir / "manuscript_metrics.json")
    update_results_json("Bayley3_Sankey_Theo_vs_Det",
        _sankey_metrics(t_exp[v], d_exp[v], theo_names, detected_names),
        filename=metrics_file)


if __name__ == "__main__":
    main()
