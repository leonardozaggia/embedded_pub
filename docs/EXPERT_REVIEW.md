# The expert review (Webappendix Table S8: item ledger and disagreements)

Two co-authors with domain expertise and no prior exposure to the Bayley
scales (A.D. and A.H.) reviewed items independently, blind to the algorithmic
solutions, from the item title, administration instructions and scoring
criteria only. The review had three targets; the same folder
(`expert_review/`) holds the scripts that produced the review workbooks, the
returned workbooks, and the scripts that compute the agreement statistics.

| Target | Role | Items | Record |
|---|---|---|---|
| **Objective 1 -- UC-flagged Bayley-III items** (sheet "UC Uncertain (Step 1)") | *decisive*: where the two experts agreed and their consensus differed from the algorithm, the consensus entered the final 91-item taxonomy | 6 (COG_027, 033, 048, 053, 065, 069) | `records/coauthor_review_b3_solution.xlsx`; final result in `results/tables/bayley3_item_domain_assignments.csv` |
| **Objective 2 -- Bayley-4 items whose unsupervised cluster differed from Aylward et al. (2022)** | *convergent-validity benchmark*, did not change any model | 16 discordant items (workbook: 17 rows, see below) | `records/coauthor_review_b4.xlsx` (pre-filled from the raw returns in `records/b4_raw/`), `records/coauthor_review_b4_kappa.json`; item-level ledger `results/tables/table_s8a_expert_review_b4.csv` (Table S8a) |
| **Objective 3 -- Bayley-III items whose bottom-up cluster differed from the translated structure** (sheet "Discordant (Step 3 vs Step 1)", 55 rows, plus the 6 UC-flagged items) | *convergent-validity benchmark* | 61 reviewed (60 discordant under the final structure) | `records/coauthor_review_b3_solution.xlsx`, `records/coauthor_review_b3_kappa.json`, `records/paired_vs_extension_*.csv`; item-level ledger `results/tables/table_s8b_expert_review_b3.csv` (Table S8b) |

Agreement was quantified with Cohen's kappa (`sklearn.metrics.cohen_kappa_score`)
on the items both raters labelled; multi-label answers were reduced to the
first-listed domain (`import_review_b3.primary_factor`).

## Objective 1: the six flagged items

| item | title | algorithm (nearest domain) | A.D. | A.H. | consensus | final |
|---|---|---|---|---|---|---|
| COG_027 | Picks Up Block Series: Reaches for Second Block | GDPS | GDPS | GDPS (or HOP) | GDPS | GDPS (confirmed) |
| COG_033 | Picks Up Block Series: Retains 2 of 3 Blocks | GDPS | ATT | ATT | ATT | **ATT** (override) |
| COG_048 | Relational Play Series: Self | GDPS | FS | GDPS | -- | GDPS (no consensus, algorithm kept) |
| COG_053 | Relational Play Series: Others | GDPS | GDPS | GDPS | GDPS | GDPS (confirmed) |
| COG_065 | Representational Play | GDPS | FS | FS | FS | **FS** (override) |
| COG_069 | Imaginary Play | GDPS | FS (HOP, FS) | FS | FS | **FS** (override) |

Distribution before / after the review: ATT 21 / 22, WM 19 / 19, GDPS 26 / 23,
FS 18 / 20, HOP 7 / 7. All three overridden items lie outside the 23-item
empirical model (COG_033 is below the dHCP window; COG_065 and COG_069 are
removed by the floor/ceiling screen), so the CFA, MIMIC and outcome results
are unaffected.

## Objective 2: Bayley-4 (16 discordant items, `coauthor_review_b4_kappa.json`)

Step 2 clustered 40 Bayley-4 items: the 39 that Aylward et al. assign to a
domain (three of them to two domains: COG_035 GDPS+FS, COG_048 GDPS+HOP,
COG_051 FS+HOP) plus COG_047, which only their empirical model contains. An
item is concordant when its cluster matches any of its expert domains
(24/40). The 16 discordant items are the 15 whose cluster matches none of
their domains and COG_047, which has none. The workbook sent to the experts
had 17 rows because a label-string mismatch in the pipeline also flagged
COG_048 (clustered HOP, one of its two domains); neither expert rated it, so
every kappa below is computed on the 16 discordant items (COG_047 enters the
"vs Aylward" comparisons as a disagreement).

| comparison | kappa | n |
|---|---|---|
| expert 1 vs expert 2 | 0.32 | 16 |
| expert 1 vs Aylward reference | 0.34 | 16 |
| expert 2 vs Aylward reference | 0.42 | 16 |
| consensus vs Aylward reference | 0.65 | 8 |
| consensus vs unsupervised cluster | -0.33 | 8 |

## Objective 3: Bayley-III (61 reviewed items, `coauthor_review_b3_kappa.json`)

"Translated structure" in this table is the algorithmic Step-1 assignment the
experts saw (before their own overrides); the kappas against the final
structure are given in the second block.

| comparison | kappa (61 reviewed) | n | kappa (60 discordant under the final structure) | n |
|---|---|---|---|---|
| A.D. vs A.H. | 0.33 | 54 | 0.32 | 53 |
| A.D. vs translated structure (pre-review) | 0.28 | 54 | 0.27 | 53 |
| A.H. vs translated structure (pre-review) | 0.20 | 61 | 0.19 | 60 |
| A.D. vs final translated structure | 0.36 | 54 | 0.35 | 53 |
| A.H. vs final translated structure | 0.26 | 61 | 0.25 | 60 |
| A.D. vs bottom-up cluster | -0.11 | 54 | -0.12 | 53 |
| A.H. vs bottom-up cluster | -0.20 | 61 | -0.22 | 60 |
| consensus vs translated (pre-review) / bottom-up | 0.40 / -0.17 | 25 | 0.39 / -0.21 | 24 |

`paired_vs_extension_agreement.py` splits the same comparisons by how the
item entered the translated structure (cascade pairing vs centroid extension),
`records/paired_vs_extension_summary.csv`.

The 61 reviewed items are the 55-row discordant sheet plus the 6 UC-flagged
items (4 of which were also discordant): 59 items were discordant under the
pre-review algorithmic structure. The experts' override of COG_033
(GDPS -> ATT) made it discordant, giving the 60 items of
`results/tables/mismatches_ledger_b3.csv`; COG_027 (UC-flagged, confirmed
GDPS, clustered GDPS) was reviewed but was never discordant. No item became
concordant through the overrides.

## Table S8: the item ledger

`results/tables/table_s8a_expert_review_b4.csv` (16 rows: Aylward domain,
detected cluster, Expert 1, Expert 2, consensus) and
`table_s8b_expert_review_b3.csv` (60 rows: algorithmic label before the
overrides, final label, bottom-up cluster, Expert 1, Expert 2, consensus)
are the item-level ledgers behind the kappas above -- item ids and domain
labels only, no item text. Expert 1 = A.D., Expert 2 = A.H.; answers are
reduced to the first-listed domain; "--" marks a blank answer or no
consensus. COG_048 (Bayley-4, unrated) and COG_027 (Bayley-III, reviewed but
concordant) are excluded. `tables/make_supplementary_tables.py` rebuilds them
when the review workbooks are present and otherwise keeps the committed CSVs,
which are frozen in `tests/fixtures/`.

## Reproducing

```bash
python expert_review/generate_review_workbook_b3.py --output /tmp/b3.xlsx   # blank template (what the experts received)
python expert_review/generate_review_workbook_b4.py --output /tmp/b4.xlsx   # pre-filled Bayley-4 workbook
python expert_review/import_review_b4.py            # kappa (Bayley-4)
python expert_review/import_review_b3.py            # kappa (Bayley-III) + model spec all_mpnet_base_v2_b3_coauthor_consensus
python expert_review/paired_vs_extension_agreement.py
python analysis/item_assignments.py                 # taxonomy table with provenance
```

The returned workbooks reproduce the item instructions and are therefore not
committed (`DATA_ACCESS.md`); the committed `*_kappa.json` / `.csv` files and
the taxonomy table carry every assignment.
