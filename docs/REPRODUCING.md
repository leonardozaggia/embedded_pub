# Reproducing the analysis

Three levels are possible, depending on what you have access to:

| Level | Needs | What you can regenerate |
|---|---|---|
| **A. Committed outputs only** | this repository, Python | all five main figures except Figure 3C's score distributions, Figures S1-S4 and S6-S8, Tables S1 and S3-S5, all concordance metrics, the 91-item taxonomy, the expert-review agreement statistics, the manuscript-number check |
| **B. + dHCP data** | controlled-access dHCP/NDA instruments (`DATA_ACCESS.md`), R | `combined.dat` and Table S2, the sample descriptives, Figure 3C, Figure S5 (tetrachoric correlations) |
| **C. + Bayley manuals + Mplus** | the item text files transcribed from the manuals, the Sentence-BERT model (downloaded automatically), a licensed Mplus 8.6 | the embedding / pairing / clustering pipeline itself (steps 1-3) and every Mplus model |

`make` targets wrap the commands below (`make help` lists them; `make all`
runs everything that is possible on the current machine and stops with a
clear message at the first missing input).

## 1. Environment

```bash
conda env create -f environment.yml      # python 3.11 + editable install of bayley_nlp
conda activate bayley-nlp
# or: pip install -r requirements.txt && pip install -e .
```

R (>= 4.3) with `dplyr`, `tidyr`, `readxl`, `readr`, `psych` is needed for
`psychometrics/prepare_data.R` and `analysis/tetrachoric.R`. Mplus 8.6 is
needed only to re-fit the models in `psychometrics/`; their `.out` files are
committed and every downstream script reads those.

Plotly Sankey diagrams (`analysis/concordance/*_sankey.py`) are rasterised
with kaleido, which needs a Chrome/Chromium; set `BROWSER_PATH` to the
executable if it is not found automatically.

## 2. Level A: rebuild every figure, table and number from the committed outputs

Run from the repository root, in this order (each step's inputs are the
committed outputs of the earlier ones):

```bash
python analysis/item_assignments.py                 # 91-item taxonomy -> results/tables/bayley3_item_domain_assignments.csv
python expert_review/import_review_b4.py            # kappa, Bayley-4 review  -> expert_review/records/
python expert_review/import_review_b3.py            # kappa, Bayley-III review; model spec in config/model_specs.yaml
python expert_review/paired_vs_extension_agreement.py

python analysis/concordance/b4_publication_heatmaps.py    # Objective 2 confusion matrices + metrics
python analysis/concordance/b4_sankey.py
python analysis/concordance/b4_item_level_analysis.py     # Figure S3, mismatches ledger
python analysis/concordance/b4_pair_level_intermediacy.py # pair-level check of the "between the two" claim
python analysis/concordance/b3_publication_heatmaps.py    # Objective 3
python analysis/concordance/b3_sankey.py
python analysis/concordance/b3_item_level_analysis.py     # Figure S4, mismatches ledger

python psychometrics/mimic/run_difftest.py --parse-only   # DIFFTEST table from the committed .out files
python psychometrics/mimic/gather_mplus_outputs.py        # STDY MIMIC effects (cross-check input)
python psychometrics/mimic/gather_std_effects.py          # STDYX/STDY effects used by Figure 4A and Table S4a
python psychometrics/outcomes/extract_outcomes_associations.py
python psychometrics/mimic/check_integrity.py             # .out vs CSV drift checks
python psychometrics/outcomes/check_integrity.py

python tables/make_supplementary_tables.py          # Tables S1-S8 -> results/tables/  (S2 needs combined.dat, level B; S8 the review workbooks, else kept)

python figures/fig2_embedding.py --export-panel-c   # Figure 2 (+ panel-C table)
python figures/fig3_cfa.py                          # Figure 3 (panel C needs fscores_measurement.dat, level B)
python figures/fig4_clinical_merged.py
python figures/fig5_validation.py
python figures/supplementary/figS1_S2_centroid_heatmaps.py
python figures/supplementary/figS6_threeway_correspondence.py
python figures/supplementary/figS7_within_vs_between.py
python figures/supplementary/figS8_threshold_sensitivity.py

python figures/fig1_export.py                       # Figure 1 SVG -> pdf + 300-dpi png (+ per-panel files); needs Playwright Chrome
python figures/fig2_embedding.py --export-panel-c --panels   # --panels: individual panels + combined copy -> results/figures/submission/
python figures/fig3_cfa.py --panels                 # (or: make submission-figures for Figures 1-5)
python figures/fig4_clinical_merged.py --panels
python figures/fig5_validation.py --panels

python supplement/make_webappendix.py               # Webappendix .docx (results/supplement/); export_pdf.ps1 makes the PDF in Word
python analysis/verify_manuscript_numbers.py        # every number reported in the manuscript vs the outputs (--manuscript PATH adds the wording checks)
pytest tests/                                       # unit tests, Mplus .inp checks, frozen-output regression
```

`figures/_umap_coords.py` rebuilds the UMAP coordinate cache behind Figure 2B
(`data/outputs/figure_inputs/umap3d_coords.csv`). The committed cache is the
one used for the manuscript; UMAP layouts can move slightly between
umap-learn versions, so only re-run it deliberately.

## 3. Level B: the dHCP-dependent steps

Place the NDA instruments under `data/raw/dhcp_txt/` and the item-level
Bayley-III responses in `data/processed/cogn_id_GA.xlsx` (see `DATA_ACCESS.md`).

```bash
Rscript psychometrics/prepare_data.R                # -> psychometrics/combined.dat, varnames.txt, factor_item_map.csv
python psychometrics/checks/check_columns.py        # combined.dat / varnames consistency
python psychometrics/checks/check_missing_data.py
python analysis/sample_descriptives.py              # Methods "Sample" paragraph -> results/metrics/sample_descriptives.json
python analysis/table1_baseline.py                  # Table 1 (baseline characteristics) -> results/tables/table1_baseline.csv (or: make table1)
python analysis/cohort_counts.py                    # cohort counts + recruitment / assessment dates -> results/metrics/cohort_counts.json
Rscript analysis/tetrachoric.R                      # -> data/outputs/tetrachoric/
python figures/supplementary/figS5_cosine_vs_tetrachoric.py
python tables/make_supplementary_tables.py          # now including Table S2
```

`analysis/growth_restriction.py` and `analysis/age_at_assessment.py` can also
be run on their own for the detailed growth-restriction and corrected-age
reports.

## 4. Level C: the NLP pipeline and the Mplus models

### Steps 1-3 (embedding, pairing, clustering)

Place the item text files under `data/raw/items_description/` (format in
`DATA_ACCESS.md`). The Sentence-BERT model `all-mpnet-base-v2` is downloaded
from the Hugging Face hub on first use (~420 MB).

```bash
python -m bayley_nlp step1      # cascade pairing + centroid assignment -> data/outputs/step1_translation/
python -m bayley_nlp step2      # consensus K-Means on the 39 Bayley-4 items + reference-model comparison
python -m bayley_nlp step3      # consensus K-Means on the 91 Bayley-III items
# or: python -m bayley_nlp all
```

Step 1 writes the translated domains of items 34-68 into
`config/model_specs.yaml` (`..._mixed_centroids`) and the pairing / assignment
CSVs that everything else reads. Steps 2 and 3 use fixed random seeds
(`random_state = run index` for the 1,000 K-Means runs), so the consensus
clusters reproduce exactly for a given set of embeddings; the embeddings
themselves are deterministic for a given model and library version. The
committed run directories (`data/outputs/step2_b4_validation/..._no_dr`,
`data/outputs/step3_b3_bottomup/..._no_dr`) are the runs behind the paper.
`python -m bayley_nlp step2 --from-cache` (or `step3 --from-cache`) re-runs
only the later sub-steps on the committed embeddings and clusters.

After step 1, rerun `python expert_review/import_review_b3.py` and
`python analysis/item_assignments.py` so the taxonomy incorporates the expert
review of the flagged items, then continue with level A.

### Mplus models

```bash
cd psychometrics                                # the measurement models read combined.dat from here
mplus measurement/model_cfa_3factor.inp
mplus measurement/model_unidimensional.inp
mplus measurement/model_cfa_3factor_difftest.inp      # nested-model test, H1 (writes difftest_3factor.dat)
mplus measurement/model_unidimensional_difftest.inp   # nested-model test, H0 (prints the difference test)
cd ..
python psychometrics/mimic/run_difftest.py      # runs every free + constrained MIMIC model and the DIFFTESTs
                                                # (MPLUS_EXE env var overrides the Mplus path)
# unidimensional MIMIC and the per-outcome models:
#   psychometrics/mimic/<cov>/model_mimic_<cov>_UNIDIM.inp
#   psychometrics/outcomes/<outcome>/model_structural_<outcome>.inp (+ _UNIDIM)
```

Then rerun the extraction scripts of level A
(`gather_std_effects.py`, `extract_outcomes_associations.py`,
`make_supplementary_tables.py`, `fig3_cfa.py`, `fig4_clinical_merged.py`).
`tests/test_reference_outputs.py` will flag any change relative to the
frozen manuscript outputs; the committed `.out` files remain the ground truth
reported in the paper.

The nested-model difference test (unidimensional vs three-factor, 3 df) is
documented in `psychometrics/measurement/README.md`.
