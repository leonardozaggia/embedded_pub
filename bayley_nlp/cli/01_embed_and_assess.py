
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Step 1: Generate Embeddings and Assess Dimensionality

This script parses cognitive item descriptions, generates embeddings using a 
sentence transformer model, and assesses the optimal dimensionality using 
multiple criteria (parallel analysis, broken stick, elbow method, BIC).

Usage:
    python 01_embed_and_assess.py --model all-mpnet-base-v2
    python 01_embed_and_assess.py --model all-roberta-large-v1 --items-file custom_items.txt

Output:
    - data/processed/embeddings_[MODEL].npy
    - data/processed/items.csv
    - data/processed/similarity_matrix_[MODEL].npy
    - data/outputs/experiments/[TIMESTAMP]/dimensionality_assessment/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bayley_nlp.core import (
    parse_items,
    load_model,
    embed_items,
    compute_similarity,
    parallel_analysis,
    broken_stick_k,
    elbow_k,
    ppca_bic_k,
    select_k_non_circular,
    ensure_dir,
)
from bayley_nlp.viz import plot_heatmap, plot_scree


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate embeddings and assess optimal dimensionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        choices=["all-mpnet-base-v2", "all-roberta-large-v1"],
        help="Sentence transformer model name",
    )
    ap.add_argument(
        "--items-file",
        type=str,
        default=None,
        help="Path to items text file (default: data/raw/items_description/b4_cog_items.txt)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/outputs/experiments/[TIMESTAMP])",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up paths
    root = Path(__file__).resolve().parents[2]
    
    if args.items_file is None:
        items_file = root / "data" / "raw" / "items_description" / "b4_cog_items.txt"
    else:
        items_file = root / "data" / "raw" / "items_description" / args.items_file

    # Create output folder name based on model
    model_slug = args.model.replace("/", "_").replace("-", "_")
    
    if args.output_dir is None:
        output_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis"
    else:
        output_dir = Path(args.output_dir)

    processed_dir = root / "data" / "processed"
    ensure_dir(str(processed_dir))
    ensure_dir(str(output_dir))

    print(f"\n{'='*70}")
    print(f"STEP 1: EMBEDDING GENERATION & DIMENSIONALITY ASSESSMENT")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"Items file: {items_file}")
    print(f"Output directory: {output_dir}\n")

    # 1. Parse items
    print("[1/5] Parsing items...")
    items = parse_items(str(items_file))
    print(f"      Parsed {len(items)} items")

    # Save items to output directory (run-specific)
    items_out = output_dir / "items.csv"
    items.to_csv(items_out, index=False)
    print(f"      Saved to: {items_out}")
    
    # Also save to processed/ for backward compatibility
    items_out_legacy = processed_dir / "items.csv"
    items.to_csv(items_out_legacy, index=False)

    # 2. Generate embeddings
    print(f"\n[2/5] Generating embeddings with {args.model}...")
    model = load_model(args.model)
    emb = embed_items(items, text_column="item_text", model=model)
    print(f"      Embedding shape: {emb.shape}")

    # Save embeddings to output directory (run-specific)
    model_slug = args.model.replace("/", "_").replace("-", "_")
    emb_out = output_dir / f"embeddings_{model_slug}.npy"
    np.save(emb_out, emb)
    print(f"      Saved to: {emb_out}")
    
    # Also save to processed/ for backward compatibility
    emb_out_legacy = processed_dir / f"embeddings_{model_slug}.npy"
    np.save(emb_out_legacy, emb)

    # 3. Compute similarity matrix
    print("\n[3/5] Computing similarity matrix...")
    S = compute_similarity(emb)

    # Save similarity matrix to output directory (run-specific)
    sim_out = output_dir / f"similarity_matrix_{model_slug}.npy"
    np.save(sim_out, S)
    print(f"      Saved to: {sim_out}")

    # 4. Assess dimensionality
    print("\n[4/5] Assessing optimal dimensionality...")
    dim_dir = ensure_dir(str(output_dir / "01_dimensionality_assessment"))

    # Eigenvalues
    ev, _ = np.linalg.eigh(S)
    ev_desc = ev[::-1]

    # Multiple criteria
    k_bs = int(broken_stick_k(ev_desc))
    k_el = int(elbow_k(ev_desc))
    k_bic = int(ppca_bic_k(emb, k_min=1, k_max=min(10, emb.shape[1] - 1)))

    # Parallel analysis
    pa_suggest, eig_obs_pa, eig_cut_pa = parallel_analysis(
        S,
        n_iter=1000,
        random_state=42,
        percentile=95.0,
        p_eff=int(emb.shape[1]),
        simulate="corr",
    )

    # Non-circular selection
    sel = select_k_non_circular(
        S,
        embed_dim=int(emb.shape[1]),
        pa_iter=1000,
        cv_kmax=min(30, S.shape[0] - 1),
        seed=42,
    )
    chosen_k_nc = int(sel["k"])

    # Legacy vote
    from collections import Counter

    votes = [k_bs, k_el, k_bic]
    cnt = Counter(votes)
    top = cnt.most_common()
    if len(top) == 0:
        chosen_k_legacy = max(1, int(k_el))
    elif len(top) == 1 or top[0][1] > top[1][1]:
        chosen_k_legacy = int(top[0][0])
    else:
        chosen_k_legacy = int(
            k_el
            if cnt[k_el] == top[0][1]
            else (k_bs if cnt[k_bs] == top[0][1] else k_bic)
        )

    # Final choice
    decision_used = "non_circular" if sel["stability"]["stable"] else "legacy"
    chosen_k = chosen_k_nc if decision_used == "non_circular" else chosen_k_legacy

    print(f"      Broken Stick: {k_bs}")
    print(f"      Elbow: {k_el}")
    print(f"      PPCA BIC: {k_bic}")
    print(f"      Parallel Analysis: {pa_suggest}")
    print(f"      Non-circular (stable): {chosen_k_nc}")
    print(f"      Legacy vote: {chosen_k_legacy}")
    print(f"      → RECOMMENDED: {chosen_k} factors (via {decision_used})")

    # Save results
    results = {
        "model": args.model,
        "timestamp": timestamp,
        "n_items": len(items),
        "embedding_dim": int(emb.shape[1]),
        "final_used_k": int(chosen_k),
        "decision_used": decision_used,
        "non_circular_selection": {
            "selected_k": int(sel["k"]),
            "candidate_k": int(sel["k_candidate"]),
            "parallel_analysis_k": int(sel["k_pa"]),
            "cv_one_se_k": int(sel["k_cv_one_se"]),
            "stability": {
                "median_phi": float(sel["stability"]["median_phi"]),
                "per_component_median": sel["stability"]["per_comp_median"],
                "stable": bool(sel["stability"]["stable"]),
            },
        },
        "legacy_vote": {
            "legacy_vote_k": int(chosen_k_legacy),
            "criteria": {
                "broken_stick": int(k_bs),
                "elbow": int(k_el),
                "ppca_bic": int(k_bic),
                "parallel_analysis_suggested": int(pa_suggest) if pa_suggest is not None else None,
            },
        },
    }
    
    results_file = dim_dir / "dimensionality_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"      Results saved to: {results_file}")

    # Save eigenvalues
    eig_df = pd.DataFrame({"eigenvalue": ev_desc})
    eig_df.to_csv(dim_dir / "eigenvalues.csv", index=False)

    # 5. Generate visualizations
    print("\n[5/5] Generating visualizations...")
    plot_heatmap(
        S,
        items["item_id"].tolist(),
        str(dim_dir / "similarity_heatmap.png"),
        figsize=(12, 10),
    )
    print(f"      Heatmap saved")

    plot_scree(
        eig_obs_pa,
        eig_cut_pa,
        str(dim_dir / "parallel_analysis_scree.png"),
        chosen_k=chosen_k,
    )
    print(f"      Scree plot saved")

    print(f"\n{'='*70}")
    print("STEP 1 COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Next step: python bayley_nlp/cli/02_factor_analysis.py --n-factors {chosen_k}")
    print(f"           (or use a different number based on your domain knowledge)\n")


if __name__ == "__main__":
    main()
