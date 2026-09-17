# Frozen reference outputs

Snapshots of the outputs behind the manuscript, used by
`tests/test_reference_outputs.py` to catch silent drift after a code change.
Each test compares one live output file with its snapshot here; the test is
skipped when the live file does not exist (for example the level-B files on
a clone without the restricted data).

| Fixture | Live file | What it freezes |
|---|---|---|
| `reference_step1_cascade_pairs.csv` | `data/outputs/step1_translation/b4_b3_cascade_pairs.csv` | the 39 cascade Bayley-4 -> Bayley-III pairs (Table S1) |
| `reference_step1_model_spec.yaml` | `data/outputs/step1_translation/mixed_centroids/b3_model_spec.yaml` | translated domains for items 34-68 (the Mplus measurement model) |
| `reference_step2_cluster_assignments.csv` | `data/outputs/step2_b4_validation/.../03_clustering/cluster_assignments.csv` | consensus clusters of the 39 Bayley-4 items (Objective 2) |
| `reference_step2_comparison_metrics.json` | `.../05_model_comparison/comparison_metrics.json` | ARI/NMI vs the expert-derived and empirical Aylward models |
| `reference_step3_cluster_assignments.csv` | `data/outputs/step3_b3_bottomup/.../03_clustering/cluster_assignments.csv` | consensus clusters of the 91 Bayley-III items (Objective 3) |
| `reference_item_domain_assignments.csv` | `results/tables/bayley3_item_domain_assignments.csv` | the final 91-item taxonomy with provenance |
| `reference_manuscript_metrics.json` | `results/metrics/manuscript_metrics.json` | concordance metrics (Figure 5, Results) |
| `reference_mismatches_ledger_b3.csv`, `_b4.csv` | `results/tables/mismatches_ledger_*.csv` | items whose bottom-up cluster differs from the reference structure |
| `reference_factor_item_map.csv` | `psychometrics/factor_item_map.csv` | item -> factor map exported by `prepare_data.R` |
| `reference_model_cfa_3factor.out` | `psychometrics/measurement/model_cfa_3factor.out` | the three-factor CFA (frozen Mplus output; cannot be regenerated without a licence) |
| `reference_model_unidimensional_difftest.out` | `psychometrics/measurement/model_unidimensional_difftest.out` | the nested-model difference test, unidimensional vs three-factor (frozen Mplus output) |
| `reference_difftest_results.csv` | `psychometrics/mimic/difftest_results.csv` | WLSMV DIFFTEST factor contrasts (Table S4b) |
| `reference_mimic_std_effects.csv` | `psychometrics/mimic/mimic_std_effects.csv` | MIMIC coefficients (Figure 4A, Table S4a) |
| `reference_outcomes_associations.csv` | `psychometrics/outcomes/outcomes_associations.csv` | factor-outcome correlations (Figure 4B, Table S5) |
| `reference_sample_descriptives.json` | `results/metrics/sample_descriptives.json` | the Methods "Sample" paragraph |
| `reference_expert_review_b3_kappa.json`, `_b4_kappa.json` | `expert_review/records/coauthor_review_*_kappa.json` | inter-rater agreement of the expert review |
| `reference_table_s6_b4_domain_recovery.csv` | `results/tables/table_s6_b4_domain_recovery.csv` | Table S6: per-domain recovery of the Aylward domains (24/39; ATT 9/9 ... HOP 1/6) |
| `reference_table_s7_b4_pair_level_intermediacy.csv` | `results/tables/table_s7_b4_pair_level_intermediacy.csv` | Table S7: pair-level intermediacy summary (287/299 kept apart, 30/39 kept together, 67/60 disputed) |
| `reference_cohort_counts.json` | `results/metrics/cohort_counts.json` | cohort counts and study dates: 984 subjects across the instruments, 739 with a Bayley-III record, 739 analysed; recruitment 2014-03-06 to 2020-10-31, 18-month assessments 2015-10-02 to 2022-10-24 (level B) |
| `reference_table1_baseline.csv` | `results/tables/table1_baseline.csv` | Table 1: baseline characteristics of the 739 infants, one row per table line (level B; the MIMIC covariate rows carry the n of `combined.dat` / the Mplus models) |
| `reference_table_s8a_expert_review_b4.csv`, `_s8b_expert_review_b3.csv` | `results/tables/table_s8a_expert_review_b4.csv`, `_s8b_...csv` | Table S8: expert-review ledger (16 / 60 discordant items); the only copy that can be rebuilt without the git-ignored review workbooks |

Numeric CSV columns are compared exactly (the pipelines are deterministic:
fixed random seeds, committed embeddings, committed Mplus outputs); the JSON
metrics are compared to 4 decimals; the `.out` files byte for byte.

Numeric CSV columns are compared exactly (the pipelines are deterministic:
fixed random seeds, committed embeddings, committed Mplus outputs); the JSON
metrics are compared to 4 decimals; the `.out` files byte for byte.

Numeric CSV columns are compared exactly (the pipelines are deterministic:
fixed random seeds, committed embeddings, committed Mplus outputs); the JSON
metrics are compared to 4 decimals; the `.out` files byte for byte.

Do not edit these files by hand. To re-capture a fixture deliberately, after
a change that is meant to alter the analysis, copy the regenerated live file
over its snapshot, for example

```bash
cp results/metrics/manuscript_metrics.json tests/fixtures/reference_manuscript_metrics.json
```

and state the reason in the commit message.
