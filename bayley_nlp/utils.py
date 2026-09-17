import json
import os
from pathlib import Path
import pandas as pd


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def update_results_json(section_name, new_data, filename="manuscript_metrics.json"):
    """
    Safely appends or updates a section in the central JSON file.
    
    Args:
        section_name (str): The key for this block of data (e.g., 'Bayley4_Sankey_Theo_vs_Det')
        new_data (dict): Dictionary containing the metrics to save.
        filename (str): The central JSON file to write to.
    """
    # Load existing data if the file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    # Update the specific section and save
    results[section_name] = new_data

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Successfully updated '{section_name}' in {filename}")


def export_mismatches_ledger(df, theoretical_col, detected_col, item_id_col, filename="mismatches_ledger.csv"):
    """
    Filters a DataFrame for items where theoretical and detected labels disagree,
    and exports a clean CSV for manuscript reporting.
    """
    # Find rows where the labels don't match
    mismatches = df[df[theoretical_col] != df[detected_col]].copy()
    
    # Keep only the essential columns for reporting
    cols_to_keep = [item_id_col, theoretical_col, detected_col]
    if 'probability' in df.columns:
        cols_to_keep.append('probability')
        
    # Sort by theoretical factor so the bleeding patterns are grouped together
    clean_mismatches = mismatches[cols_to_keep].sort_values(by=theoretical_col)
    
    clean_mismatches.to_csv(filename, index=False)
    print(f"✅ Exported {len(clean_mismatches)} mismatched items to {filename}")