# From Composite to Components: domain-differentiated scoring of the Bayley-III cognition scale

Code, data products and reproduction workflow for

> Zaggia L, Diederich A, Hilgendorff A, Heep A, Hildebrandt A. *From Composite
> to Components: Evaluating the Impact of Domain Differentiation on Bayley-III
> Cognitive Diagnostic Precision.* Submitted to *The Lancet Child & Adolescent
> Health*.

The paper translates the five-domain cognitive structure of the Bayley-4
(Aylward et al., 2022) onto all 91 Bayley-III cognition items with
Sentence-BERT item embeddings, validates the resulting structure
psychometrically in 739 infants of the developing Human Connectome Project
(CFA, MIMIC and outcome models), and checks it against fully unsupervised
consensus clustering.

## What the repository contains

* the analysis code: the embedding / pairing / clustering pipeline
  (`bayley_nlp/`), the Mplus models with their outputs (`psychometrics/`),
  the expert-review scripts and record (`expert_review/`), the concordance
  and sample analyses (`analysis/`), and the scripts that draw every figure
  and build every table (`figures/`, `tables/`, `supplement/`);
* the committed intermediate outputs that make the paper reproducible
  without the restricted inputs: item embeddings, cascade pairs, consensus
  clusters, Mplus `.out` files and extracted coefficient tables;
* the results as they appear in the paper and its supplement
  (`results/`), including the openly shared item-to-domain assignments of
  all 91 Bayley-III cognition items
  (`results/tables/bayley3_item_domain_assignments.csv`) and the
  expert-review ledger (`results/tables/table_s8*.csv`).

```
bayley_nlp/        Python package: item parsing, embedding, cascade pairing, centroid
                   assignment (step 1), consensus K-Means + labelling + reference-model
                   comparison (steps 2-3).  Entry point: python -m bayley_nlp {step1,step2,step3,all}
config/            reference models of Aylward et al. (2022), domain definitions, model specs
data/              raw/ (restricted inputs, not distributed), processed/, outputs/ (pipeline runs)
psychometrics/     dHCP data preparation (R), Mplus models (.inp + .out) and extraction scripts:
                   measurement/ (CFA), mimic/ (18 covariates + DIFFTEST), outcomes/ (16 outcomes)
expert_review/     the co-author review: workbook generators, agreement statistics, records/
analysis/          sample descriptives, the 91-item taxonomy, concordance metrics (Objectives 2-3),
                   tetrachoric correlations, manuscript-number verification
figures/           Figures 1-5 (fig*.py) and supplementary/ Figures S1-S8, shared style module
tables/            Supplementary Tables S1-S8
supplement/        builder of the Webappendix document
results/           what the paper shows: figures/main, figures/supplementary, tables, metrics, supplement
docs/              REPRODUCING.md, DATA_ACCESS.md, EXPERT_REVIEW.md
tests/             unit tests, Mplus input checks, frozen-output regression, manuscript-number check
```

## Reproducing

```bash
conda env create -f environment.yml && conda activate bayley-nlp   # or pip install -r requirements.txt && pip install -e .
make results        # rebuild every figure, table and metric from the committed outputs
make verify         # check every number reported in the manuscript against the outputs, then run the test suite
```

`make help` lists the individual targets. Three levels of reproduction are
possible, depending on what you have access to; `docs/REPRODUCING.md` gives
the commands for each.

| Level | Needs | Regenerates |
|---|---|---|
| A | this repository, Python | every figure, table and reported statistic, from the committed outputs |
| B | + the controlled-access dHCP data, R | the Mplus analysis file, the sample descriptives, Table 1, Table S2, Figure 3C, Figure S5 |
| C | + the Bayley item text, Mplus 8.6 | the embedding / pairing / clustering pipeline itself and every Mplus model |

## Restricted data

Two inputs are not distributed (`docs/DATA_ACCESS.md`):

* **Bayley item text.** The administration instructions and scoring
  criteria of the Bayley-III and Bayley-4 cognition items are copyrighted
  test material. Item titles appear in the committed tables, as they do in
  the paper; the full text is needed only to re-run the embedding step and
  has to be transcribed from the published manuals.
* **dHCP clinical data.** The developing Human Connectome Project
  instruments (fourth release) are held in the NIMH Data Archive, collection
  3955 (<https://nda.nih.gov/edit_collection.html?id=3955>), and are available
  to investigators with an approved data-access request.

Everything derived from these inputs that contains neither individual-level
data nor item wording is committed, so that level A needs nothing beyond
this repository.

## The analysis in one paragraph

1. **Objective 1, translation.** Item titles, instructions and scoring
   criteria of the 39 Bayley-4 items with an expert domain assignment and of
   all 91 Bayley-III items are embedded with `all-mpnet-base-v2`. Each
   Bayley-4 item is paired with a Bayley-III item by cascade matching
   (title cosine >= .75 locks a pair, otherwise full-text similarity, greedy
   and exclusive); the 52 unpaired Bayley-III items are assigned to the
   nearest domain centroid, with items closer to an unclassified cluster by
   more than .10 referred to two blind expert reviewers
   (`expert_review/`). Result: ATT 22, WM 19, GDPS 23, FS 20, HOP 7 items.
2. **Objective 1, psychometric validation.** Items 34-68 (the dHCP window)
   retained after a 90 % floor/ceiling screen (23 items in WM, GDPS, FS) are
   modelled in Mplus (WLSMV): a three-factor CFA vs a unidimensional model
   (nested-model difference test), 18 MIMIC covariate models with WLSMV difference tests of the factor
   contrasts, and 16 outcome models (`psychometrics/`).
3. **Objective 2.** Consensus K-Means (1,000 runs) on the 39 Bayley-4 items,
   labelled post hoc, compared with Aylward's expert-derived and empirical
   models (ARI/NMI, Sankey), plus expert review of the discordant items.
4. **Objective 3.** The same pipeline on all 91 Bayley-III items, compared
   with the translated structure.

An interactive explorer of the item assignments is at
<https://leonardozaggia.github.io/bayley-cognitive-map/>.

## Licence

MIT for the code (`LICENSE`). The Bayley item text and the dHCP data are not
part of this repository and remain under their own terms.

## Citation

Please cite the paper above; `CITATION.cff` carries the reference in
machine-readable form.
