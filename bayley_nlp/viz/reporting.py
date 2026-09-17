from __future__ import annotations

import os
from typing import Dict, List, Optional
from bayley_nlp.utils import ensure_dir


def summary_report(
    outdir: str,
    title: str,
    selection: Dict[str, object],
    diagnostics: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> str:
    """Write a concise Markdown report with key metrics and settings.

    Returns the path to the markdown file saved under outdir.
    """
    ensure_dir(outdir)
    md_path = os.path.join(outdir, "summary.md")

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Selected Parameters")
    lines.append(f"- min_cluster_size: {selection.get('min_cluster_size')}")
    lines.append(f"- min_samples: {selection.get('min_samples')}")
    if diagnostics.get("used_umap"):
        uk = diagnostics.get("umap_kwargs", {})
        lines.append("- UMAP: yes")
        lines.append(
            f"  - metric={uk.get('metric')}, n_neighbors={uk.get('n_neighbors')}, "
            f"min_dist={uk.get('min_dist')}, n_components={uk.get('n_components')}"
        )
    else:
        lines.append("- UMAP: no (raw embedding space)")

    lines.append("")
    lines.append("## Diagnostics")
    lines.append(f"- DBCV: {diagnostics.get('dbcv')}")
    lines.append(f"- Silhouette (masked): {diagnostics.get('silhouette')}")
    lines.append(f"- Noise rate: {diagnostics.get('noise_rate')}")
    sizes = diagnostics.get("cluster_sizes", {})
    lines.append(f"- Cluster sizes: {sizes}")

    if extra:
        lines.append("")
        lines.append("## Extra")
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path

