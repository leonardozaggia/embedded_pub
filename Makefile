# Reproduction targets. Run from the repository root; `make help` lists them.
# Every target runs the commands listed in docs/REPRODUCING.md, in order.

PY      ?= python
RSCRIPT ?= Rscript
MPLUS   ?= mplus

.DEFAULT_GOAL := help
.PHONY: help all results taxonomy expert-review concordance psychometrics-extract tables figures \
        supplementary-figures submission-figures supplement verify test dhcp table1 nlp mplus clean-results

help:
	@echo "make results        rebuild figures, tables and metrics from the committed outputs (level A)"
	@echo "make verify         check all manuscript numbers + run the test suite"
	@echo "make dhcp           level B: combined.dat, sample descriptives, Table 1, cohort counts, tetrachoric (needs dHCP data + R)"
	@echo "make table1         level B: Table 1 (baseline characteristics) -> results/tables/table1_baseline.csv"
	@echo "make submission-figures  Figures 1-5 as combined + individual-panel files (pdf/svg/png) -> results/figures/submission/ (needs Playwright Chrome for Figure 1)"
	@echo "make nlp            level C: python -m bayley_nlp all (needs item text + model), then taxonomy"
	@echo "make mplus          level C: re-fit the Mplus models (needs Mplus; MPLUS=path/to/mplus)"
	@echo "make all            dhcp + results + verify"
	@echo "make supplement     build the Webappendix .docx (then export_pdf.ps1 in Word for the PDF)"
	@echo "sub-targets: taxonomy expert-review concordance psychometrics-extract tables figures supplementary-figures"

all: dhcp results verify

# ---------------------------------------------------------------- level A
results: taxonomy expert-review concordance psychometrics-extract tables figures supplementary-figures

taxonomy:
	$(PY) analysis/item_assignments.py

expert-review:
	$(PY) expert_review/import_review_b4.py
	$(PY) expert_review/import_review_b3.py
	$(PY) expert_review/paired_vs_extension_agreement.py

concordance: taxonomy
	$(PY) analysis/concordance/b4_publication_heatmaps.py
	$(PY) analysis/concordance/b4_sankey.py
	$(PY) analysis/concordance/b4_item_level_analysis.py
	$(PY) analysis/concordance/b4_pair_level_intermediacy.py
	$(PY) analysis/concordance/b3_publication_heatmaps.py
	$(PY) analysis/concordance/b3_sankey.py
	$(PY) analysis/concordance/b3_item_level_analysis.py

psychometrics-extract:
	$(PY) psychometrics/mimic/run_difftest.py --parse-only
	$(PY) psychometrics/mimic/gather_mplus_outputs.py
	$(PY) psychometrics/mimic/gather_std_effects.py
	$(PY) psychometrics/outcomes/extract_outcomes_associations.py
	$(PY) psychometrics/mimic/check_integrity.py
	$(PY) psychometrics/outcomes/check_integrity.py

tables: taxonomy psychometrics-extract
	$(PY) tables/make_supplementary_tables.py

figures: taxonomy concordance psychometrics-extract
	$(PY) figures/fig2_embedding.py --export-panel-c
	$(PY) figures/fig3_cfa.py
	$(PY) figures/fig4_clinical_merged.py
	$(PY) figures/fig5_validation.py

supplementary-figures: taxonomy concordance
	$(PY) figures/supplementary/figS1_S2_centroid_heatmaps.py
	$(PY) figures/supplementary/figS6_threeway_correspondence.py
	$(PY) figures/supplementary/figS7_within_vs_between.py
	$(PY) figures/supplementary/figS8_threshold_sensitivity.py
	@test -f data/outputs/tetrachoric/tetrachoric_matrix.csv && $(PY) figures/supplementary/figS5_cosine_vs_tetrachoric.py \
	  || echo "Figure S5 skipped: run 'make dhcp' first (needs data/outputs/tetrachoric/)"

# The journal asks for multi-part figures as the individual parts plus a combined
# version, in editable vector form.
submission-figures: taxonomy concordance psychometrics-extract
	$(PY) figures/fig1_export.py
	$(PY) figures/fig2_embedding.py --export-panel-c --panels
	$(PY) figures/fig3_cfa.py --panels
	$(PY) figures/fig4_clinical_merged.py --panels
	$(PY) figures/fig5_validation.py --panels

supplement: tables supplementary-figures
	$(PY) supplement/make_webappendix.py
	@echo "PDF: powershell -ExecutionPolicy Bypass -File supplement/export_pdf.ps1 (needs Word)"

verify:
	$(PY) analysis/verify_manuscript_numbers.py
	$(PY) -m pytest tests/ -q

test:
	$(PY) -m pytest tests/ -q

# ---------------------------------------------------------------- level B
dhcp:
	$(RSCRIPT) psychometrics/prepare_data.R
	$(PY) psychometrics/checks/check_columns.py
	$(PY) psychometrics/checks/check_missing_data.py
	$(PY) analysis/sample_descriptives.py
	$(PY) analysis/table1_baseline.py
	$(PY) analysis/cohort_counts.py
	$(RSCRIPT) analysis/tetrachoric.R
	$(PY) figures/supplementary/figS5_cosine_vs_tetrachoric.py

table1:
	$(PY) analysis/table1_baseline.py

# ---------------------------------------------------------------- level C
nlp:
	$(PY) -m bayley_nlp all
	$(PY) expert_review/import_review_b3.py
	$(PY) analysis/item_assignments.py

mplus:
	cd psychometrics && $(MPLUS) measurement/model_cfa_3factor.inp && $(MPLUS) measurement/model_unidimensional.inp
	cd psychometrics && $(MPLUS) measurement/model_cfa_3factor_difftest.inp && $(MPLUS) measurement/model_unidimensional_difftest.inp
	MPLUS_EXE=$(MPLUS) $(PY) psychometrics/mimic/run_difftest.py
	@echo "Now run the *_UNIDIM.inp MIMIC models and the per-outcome models (see docs/REPRODUCING.md), then 'make results'"

clean-results:
	rm -rf results/figures results/metrics/manuscript_metrics.json results/metrics/manuscript_numbers_check.json
