# Data access and what is (not) in this repository

Two kinds of input cannot be redistributed and are therefore excluded from git
(`.gitignore`); everything derived from them that contains no individual-level
data and no item wording *is* committed, so that every figure, table and
number can be regenerated without them (level A in `REPRODUCING.md`).

## Bayley item text (copyright)

`data/raw/items_description/`

| file | content |
|---|---|
| `b3_cog_items.txt` | the 91 Bayley-III cognition items |
| `b3_cog_items_34_68.txt` | the subset administered in the dHCP window |
| `b4_cog_items.txt` | the 81 Bayley-4 cognition items |
| `paper_items_b4.txt` | the 39 Bayley-4 items assigned to domains by Aylward et al. (2022) |

Each item is transcribed from the administration manual as

```
Item 34 - <Title>
Material: ...
<administration instructions>
1 point: <scoring criterion>
0 points: ...
```

(`bayley_nlp/core/parsing.py` documents the exact parsing rules). The Bayley
scales are commercial test instruments; the text is required only to re-run
the embedding step (level C). Item *titles* appear in the committed tables,
as they do in the paper and in the published literature.

Files that contain the wording and are therefore ignored by git: every
`items.csv` in `data/`, `data/processed/items.csv`, and the expert-review
workbooks in `expert_review/records/` (they reproduce the instructions for
the reviewers). The reviewers' assignments themselves are committed in the
`*_kappa.json`, `paired_vs_extension_*.csv` files and in
`results/tables/bayley3_item_domain_assignments.csv`.

The item embeddings (768-dimensional vectors) *are* committed
(`data/outputs/step*/**/embeddings_all_mpnet_base_v2.npy`,
`data/processed/embeddings_all_mpnet_base_v2.npy`); they do not allow the
text to be recovered and they are what the clustering and the similarity
figures read.

## dHCP clinical data (controlled access)

`data/raw/dhcp_txt/` holds the NDA-format tab-delimited instruments of the
developing Human Connectome Project (fourth release, NIMH Data Archive
collection 3955): `bsid_iii01.txt`, `lpb01.txt`, `cpenr01.txt`,
`cbcl1_501.txt`, `ecbq01.txt`, `epds01.txt`, `qucht01.txt`, `stps01.txt`,
`pqmf01.txt`, `ndar_subject01.txt`, `nnsi01.txt`, `nicu101.txt`.
`data/processed/cogn_id_GA.xlsx` holds the item-level Bayley-III cognition
responses of the 739 infants (one column per item, `bsid_cog<n>`) together
with `src_subject_id`, sex and gestational age; it is the input of
`psychometrics/prepare_data.R`.

Access is granted through the NDA data-access procedure
(<https://nda.nih.gov/edit_collection.html?id=3955>). Individual-level files
derived from these data are ignored by git as well: `psychometrics/combined.dat`
and the Mplus factor-score files `psychometrics/fscores_*.dat`,
`psychometrics/mimic/*/fscores_*.dat`, `data/outputs/growth_restriction_per_infant.csv`.

The Mplus `.out` files are committed: they contain only model results and
aggregate sample statistics.

## Committed derived data

| path | content |
|---|---|
| `data/outputs/step1_translation/` | cascade pairs, paired / unpaired assignments, model specs, centroid similarity profiles, the pipeline's diagnostic figures |
| `data/outputs/step2_b4_validation/..._no_dr/` | Objective 2 run: embeddings, consensus clusters, stability diagnostics, post-hoc labels, reference-model comparison |
| `data/outputs/step3_b3_bottomup/..._no_dr/` | Objective 3 run (same layout) |
| `data/outputs/concordance/` | confusion-matrix and Sankey intermediates of `analysis/concordance/` |
| `data/outputs/figure_inputs/` | UMAP coordinate cache (Figure 2B), title/full-text similarity matrices (Figure S8) |
| `data/outputs/tetrachoric/` | tetrachoric correlations of the 23 CFA items (Figure S5) |
| `psychometrics/**/*.inp`, `*.out`, `*.csv` | every Mplus model with its output and the extracted coefficient tables |
| `expert_review/records/` | the expert-review record (see `EXPERT_REVIEW.md`) |
| `results/` | the figures, tables and metrics as they appear in the paper and supplement |
