# Measurement models (Objective 1, psychometric validation)

| File | Model | Reported in |
|---|---|---|
| `model_cfa_3factor.inp` / `.out` | three-factor CFA (WM 4, GDPS 8, FS 11 items; 23 binary indicators, WLSMV, probit link), EAP factor scores saved to `../fscores_measurement.dat` | Figure 3, Table S3a-c, Results |
| `model_unidimensional.inp` / `.out` | unidimensional CFA on the same 23 items | Figure 3D, Table S3c-d, Results |
| `model_cfa_3factor_difftest.inp` / `.out` | H1 of the nested-model test: identical to the three-factor model (minus the factor-score output), saves the DIFFTEST derivatives `../difftest_3factor.dat` | Results (difference test), Table S3c |
| `model_unidimensional_difftest.inp` / `.out` | H0 of the nested-model test: identical to the unidimensional model, `ANALYSIS: DIFFTEST = "difftest_3factor.dat"`; the test is printed under "Chi-Square Test for Difference Testing" | Results (difference test), Table S3c |

All four read `combined.dat` (built by `../prepare_data.R`; restricted dHCP
data, not distributed) and are run with Mplus 8.6 **from the `psychometrics/`
folder**, so that the data path and the SAVEDATA paths resolve:

```
cd psychometrics
mplus measurement/model_cfa_3factor.inp
mplus measurement/model_unidimensional.inp
mplus measurement/model_cfa_3factor_difftest.inp        # writes difftest_3factor.dat
mplus measurement/model_unidimensional_difftest.inp     # consumes it
```

The `.out` files committed here are the frozen results behind the manuscript
(`tests/test_reference_outputs.py` checks that they are unchanged);
`tables/make_supplementary_tables.py` parses them into Table S3.

## Nested-model difference test (Results, unidimensional model)

The unidimensional model is the three-factor model with the three
inter-factor correlations fixed to 1, so the two models are nested (3 df) and
the WLSMV-corrected chi-square difference test applies (Mplus DIFFTEST,
Asparouhov & Muthén 2006; the same procedure as the factor contrasts in
`../mimic/run_difftest.py`). Mplus 8.6, N = 739, 23 items:

| Model | chi-square (WLSMV) | df | RMSEA | CFI | TLI | SRMR |
|---|---|---|---|---|---|---|
| three-factor (H1) | 820.910 | 227 | 0.060 | 0.929 | 0.921 | 0.108 |
| unidimensional (H0) | 1193.599 | 230 | 0.075 | 0.885 | 0.874 | 0.113 |
| **difference test** | **246.109** | **3** | p < 0.0001 | | | |

The single-model fits in the two `_difftest` outputs are identical to those
of `model_cfa_3factor.out` and `model_unidimensional.out`. The data path
echoed at the top of the two `_difftest` `.out` files
(`../../psychometrics/combined.dat`) differs from the `FILE = "combined.dat"`
of the `.inp` files; both point to the same `combined.dat`, and the `.inp`
files reproduce the outputs when run from `psychometrics/` as above.
