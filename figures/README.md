# Figures

`fig2_embedding.py` ... `fig5_validation.py` draw Figures 2-5 into
`results/figures/main/` (PDF + SVG with editable text, PNG at 400 dpi; sized
to Lancet column widths, 183 mm double / 89 mm single).
`supplementary/figS*.py` draw Figures S1, S2 and S5-S8 into
`results/figures/supplementary/` (S3 and S4 are drawn by
`analysis/concordance/*_item_level_analysis.py`). Figure 1 is a hand-drawn
SVG (`results/figures/main/fig1_pipeline.svg`); `fig1_export.py` renders it
to PDF and PNG and splits it into the per-panel submission files. Figure 1 is a hand-drawn
SVG (`results/figures/main/fig1_pipeline.svg`); `fig1_export.py` renders it
to PDF and PNG and splits it into the per-panel submission files. Figure 1 is a hand-drawn
SVG (`results/figures/main/fig1_pipeline.svg`); `fig1_export.py` renders it
to PDF and PNG and splits it into the per-panel submission files.

Shared modules:

| module | provides |
|---|---|
| `lancet_style.py` | the locked factor palette (WM `#2D5FA3`, FS `#D4783A`, GDPS `#2E9E8A`, ATT `#B5404E`, HOP `#7755A8`), typography, mid-line decimals, panel labels, native vector Sankey, `save_figure` |
| `_embedding.py` | raincloud / UMAP / within-between drawers; `translated_short_map()` and `detected_short_map()`, the item -> domain maps used by every figure |
| `_clinical.py` | loaders + dot-plot drawer for Figure 4 |
| `_validation.py` | confusion-matrix construction and drawer for Figure 5 and S6 |
| `_umap_helpers.py`, `_umap_coords.py` | UMAP parameters and the coordinate cache behind Figure 2B |

Each script's docstring lists its inputs and outputs; the run order is in
`docs/REPRODUCING.md` (or `make figures`).
