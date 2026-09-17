#!/usr/bin/env python
"""
Check every statistic reported in the manuscript against the regenerated outputs.

The expected values below are transcribed from the final manuscript text
("From Composite to Components: Evaluating the Impact of Domain Differentiation
on Bayley-III Cognitive Diagnostic Precision"); each check names the output file
the number is read from.  Run after the full pipeline (see docs/REPRODUCING.md):

    python analysis/verify_manuscript_numbers.py
    python analysis/verify_manuscript_numbers.py --manuscript path/to/manuscript.docx

Without --manuscript the script checks the regenerated outputs against the
transcribed values.  With a manuscript (.docx) it additionally checks that the
final wording is present in the text itself (24/39, 61.5 %; 40 items
clustered; 16 / 60 discordant items; the paternal-education sentence; ...).
The default location is the git-ignored folder manuscript/ inside this
repository; nothing outside the repository is ever read, so a missing
manuscript skips these wording checks with a message instead of silently
reading a draft elsewhere.

Exit status 0 = every reported number is reproduced (within the stated
rounding), 1 = at least one mismatch.  The same checks run under pytest via
tests/test_manuscript_numbers.py.  A JSON copy of the comparison is written to
results/metrics/manuscript_numbers_check.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MANUSCRIPT_DIR = ROOT / "manuscript"          # git-ignored; see manuscript/README.md
sys.path.insert(0, str(ROOT / "figures"))

CHECKS: list[dict] = []


def check(section: str, label: str, expected, observed, tol: float = 0.0, source: str = "") -> None:
    if isinstance(expected, str):
        ok = str(observed) == expected
    else:
        ok = observed is not None and abs(float(observed) - float(expected)) <= tol + 1e-12
    CHECKS.append({"section": section, "label": label, "expected": expected, "observed": observed,
                   "ok": bool(ok), "source": source})


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ---------------------------------------------------------------------------
def check_sample() -> None:
    s = json.loads((RES / "metrics/sample_descriptives.json").read_text())
    src = "results/metrics/sample_descriptives.json (analysis/sample_descriptives.py)"
    S = "Sample"
    check(S, "N infants", 739, s["n_infants"], source=src)
    check(S, "% female", 46.8, s["female"]["pct"], 0.05, src)
    ga = s["gestational_age_weeks"]
    check(S, "GA mean", 38.3, ga["mean"], 0.05, src); check(S, "GA SD", 3.8, ga["sd"], 0.05, src)
    check(S, "GA min", 23, ga["min"], 0.5, src); check(S, "GA max", 43, ga["max"], 0.5, src)
    check(S, "preterm n", 158, s["preterm_lt37"]["n"], source=src)
    check(S, "preterm %", 21.4, s["preterm_lt37"]["pct"], 0.05, src)
    check(S, "<32 wk n", 69, s["very_preterm_lt32"]["n"], source=src)
    check(S, "<32 wk %", 9.3, s["very_preterm_lt32"]["pct"], 0.05, src)
    bw = s["birth_weight_lt1500g"]
    check(S, "<1500 g n", 60, bw["n"], source=src); check(S, "<1500 g denominator", 694, bw["of"], source=src)
    check(S, "<1500 g %", 8.6, bw["pct"], 0.05, src)
    g = s["growth_restriction_sga10"]
    check(S, "growth restriction n", 81, g["n"], source=src); check(S, "growth restriction denominator", 691, g["of"], source=src)
    check(S, "growth restriction %", 11.7, g["pct"], 0.05, src)
    check(S, "growth-restricted and <1500 g", 13, g["of_whom_lt1500g"], source=src)
    b = s["birth_weight_kg"]
    check(S, "birth weight mean (kg)", 3.01, b["mean"], 0.005, src)
    check(S, "birth weight SD (kg)", 0.87, b["sd"], 0.005, src)
    check(S, "birth weight n with data", 694, b["n"], source=src)
    check(S, "birth weight min", 0.45, b["min"], 0.005, src); check(S, "birth weight max", 4.75, b["max"], 0.005, src)
    ni = s["nicu_admission"]
    check(S, "NICU admission n", 166, ni["n"], source=src)
    check(S, "NICU admission denominator (infants with data)", 694, ni["n_with_data"], source=src)
    # "166 infants (22.5 % of those with data)": 22.5 % is 166/739 (all infants);
    # of the 694 with a NICU field it is 23.9 %.  Both are frozen here.
    check(S, "NICU admission % of all 739", 22.5, ni["pct_of_all"], 0.05, src)
    check(S, "NICU admission % of the 694 with data", 23.9, ni["pct_of_those_with_data"], 0.05, src)
    p = s["plurality"]
    check(S, "singletons %", 88.9, p["singleton"]["pct"], 0.05, src)
    check(S, "twins %", 10.4, p["twin"]["pct"], 0.05, src)
    check(S, "higher-order %", 0.4, p["higher_order"]["pct"], 0.05, src)
    a = s["age_at_assessment_months"]
    check(S, "corrected age mean", 19.4, a["corrected"]["mean"], 0.05, src)
    check(S, "corrected age SD", 2.4, a["corrected"]["sd"], 0.05, src)
    check(S, "corrected age min", 17.0, a["corrected"]["min"], 0.05, src)
    check(S, "corrected age max", 34.2, a["corrected"]["max"], 0.05, src)
    check(S, "chronological age mean", 19.8, a["chronological"]["mean"], 0.05, src)
    check(S, "chronological age SD", 2.6, a["chronological"]["sd"], 0.05, src)
    check(S, "chronological age max", 37, a["chronological"]["max"], 0.5, src)
    c = s["bayley_composites"]
    for k, m, sd in (("cognitive", 100.8, 11.7), ("language", 98.7, 16.2), ("motor", 101.6, 10.1)):
        check(S, f"{k} composite mean", m, c[k]["mean"], 0.05, src)
        check(S, f"{k} composite SD", sd, c[k]["sd"], 0.05, src)
    m = s["maternal_age_years"]
    check(S, "maternal age mean", 34.2, m["mean"], 0.05, src); check(S, "maternal age SD", 4.6, m["sd"], 0.05, src)
    check(S, "maternal age range", "17-52", f"{m['min']:.0f}-{m['max']:.0f}", source=src)
    e = s["maternal_age_left_education_years"]
    check(S, "left education mean", 23.3, e["mean"], 0.05, src); check(S, "left education SD", 4.4, e["sd"], 0.05, src)
    check(S, "left education range", "12-41", f"{e['min']:.0f}-{e['max']:.0f}", source=src)


def check_table1() -> None:
    """Table 1 (results/tables/table1_baseline.csv, analysis/table1_baseline.py)
    must reproduce the Participants paragraph and carry, for every MIMIC
    covariate, the sample size of the corresponding Mplus model."""
    S = "Table 1"
    src = "results/tables/table1_baseline.csv (analysis/table1_baseline.py)"
    t = pd.read_csv(RES / "tables/table1_baseline.csv")
    t["category"] = t["category"].fillna("").astype(str)
    ix = t.set_index(["variable", "category"])

    def row(var, cat=""):
        return ix.loc[(var, cat)]

    check(S, "rows", 33, len(t), source=src)
    check(S, "N infants", 739, int(row("Sex", "female")["n_with_data"]), source=src)
    check(S, "% female", 46.8, row("Sex", "female")["sd_or_pct"], 0.05, src)
    ga = row("Gestational age at birth, weeks")
    check(S, "GA mean", 38.3, ga["value"], 0.05, src); check(S, "GA SD", 3.8, ga["sd_or_pct"], 0.05, src)
    check(S, "GA range", "23-43", f"{ga['low']:.0f}-{ga['high']:.0f}", source=src)
    pt, vpt = row("Preterm birth", "<37 weeks"), row("Very preterm birth", "<32 weeks")
    check(S, "preterm n", 158, pt["value"], source=src); check(S, "preterm %", 21.4, pt["sd_or_pct"], 0.05, src)
    check(S, "<32 wk n", 69, vpt["value"], source=src); check(S, "<32 wk %", 9.3, vpt["sd_or_pct"], 0.05, src)
    vl = row("Birth weight <1500 g", "<1500 g")
    check(S, "<1500 g n/denominator", "60/694", f"{int(vl['value'])}/{int(vl['n_with_data'])}", source=src)
    check(S, "<1500 g %", 8.6, vl["sd_or_pct"], 0.05, src)
    g = row("Growth restriction, birth weight <10th INTERGROWTH-21st centile", "<10th centile")
    check(S, "growth restriction n/denominator", "81/691", f"{int(g['value'])}/{int(g['n_with_data'])}", source=src)
    check(S, "growth restriction %", 11.7, g["sd_or_pct"], 0.05, src)
    bw = row("Birth weight, kg")
    check(S, "birth weight mean (kg)", 3.01, bw["value"], 0.005, src)
    # the birth-weight row keeps 4 decimals (SD 0.8747), so 2-decimal formatting gives the text's 0.87
    check(S, "birth weight SD (kg)", 0.87, bw["sd_or_pct"], 0.005, src)
    check(S, "birth weight n with data", 694, bw["n_with_data"], source=src)
    check(S, "birth weight min", 0.45, bw["low"], 0.005, src); check(S, "birth weight max", 4.75, bw["high"], 0.005, src)
    ni = row("NICU admission", "yes")
    check(S, "NICU admission n", 166, ni["value"], source=src)
    check(S, "NICU admission denominator (infants with data)", 694, ni["n_with_data"], source=src)
    check(S, "NICU admission % of all 739 (manuscript)", 22.5, 100 * ni["value"] / 739, 0.05, src)
    check(S, "NICU admission % of the 694 with data (Table 1)", 23.9, ni["sd_or_pct"], 0.05, src)
    pl = {c: row("Plurality", c) for c in ("singleton", "twin", "higher-order")}
    check(S, "plurality denominator (infants with data)", 737, pl["singleton"]["n_with_data"], source=src)
    check(S, "singletons % of all 739 (manuscript)", 88.9, 100 * pl["singleton"]["value"] / 739, 0.05, src)
    check(S, "singletons % of the 737 with data (Table 1)", 89.1, pl["singleton"]["sd_or_pct"], 0.05, src)
    check(S, "twins %", 10.4, pl["twin"]["sd_or_pct"], 0.05, src)
    check(S, "higher-order %", 0.4, pl["higher-order"]["sd_or_pct"], 0.05, src)
    ca, ch = row("Corrected age at assessment, months"), row("Chronological age at assessment, months")
    check(S, "corrected age mean", 19.4, ca["value"], 0.05, src); check(S, "corrected age SD", 2.4, ca["sd_or_pct"], 0.05, src)
    check(S, "corrected age min", 17.0, ca["low"], 0.05, src); check(S, "corrected age max", 34.2, ca["high"], 0.05, src)
    check(S, "chronological age mean", 19.8, ch["value"], 0.05, src); check(S, "chronological age SD", 2.6, ch["sd_or_pct"], 0.05, src)
    check(S, "chronological age range", "17-37", f"{ch['low']:.0f}-{ch['high']:.0f}", source=src)
    for k, m, sd in (("cognitive", 100.8, 11.7), ("language", 98.7, 16.2), ("motor", 101.6, 10.1)):
        c = row(f"Bayley-III {k} composite score")
        check(S, f"{k} composite mean", m, c["value"], 0.05, src); check(S, f"{k} composite SD", sd, c["sd_or_pct"], 0.05, src)
    ma = row("Maternal age at birth, years")
    check(S, "maternal age mean", 34.2, ma["value"], 0.05, src); check(S, "maternal age SD", 4.6, ma["sd_or_pct"], 0.05, src)
    check(S, "maternal age range", "17-52", f"{ma['low']:.0f}-{ma['high']:.0f}", source=src)
    me = row("Maternal age at leaving formal education, years")
    check(S, "left education mean", 23.3, me["value"], 0.05, src); check(S, "left education SD", 4.4, me["sd_or_pct"], 0.05, src)
    check(S, "left education range", "12-41", f"{me['low']:.0f}-{me['high']:.0f}", source=src)

    # every MIMIC covariate: Table 1 n_with_data == "Number of observations" of its Mplus model
    mimic_rows = {
        "GA": ("Gestational age at birth, weeks", ""), "BirthWt": ("Birth weight, kg", ""),
        "Sex": ("Sex", "female"), "HdCirc": ("Head circumference at birth, cm", ""),
        "IUGR": ("Clinically suspected intrauterine growth restriction", "yes"),
        "GDiab": ("Gestational diabetes", "yes"),
        "PrEcl": ("Pre-eclampsia", "yes"),
        "AntCor": ("Antenatal corticosteroids", "complete course"),
        "MomAge": ("Maternal age at birth, years", ""),
        "MomEdu": ("Maternal age at leaving formal education, years", ""),
        "DadEdu": ("Paternal age at leaving formal education, years", ""),
        "MomEng": ("Maternal English as first language", "yes"),
        "EPDSTot": ("Edinburgh Postnatal Depression Scale total", ""),
        "StimEnv": ("Cognitively Stimulating Parenting Scale total", ""),
        "PrLax": ("Parenting Scale, laxness", ""), "PrOvr": ("Parenting Scale, overreactivity", ""),
        "PrVrb": ("Parenting Scale, verbosity", ""), "PrSty": ("Parenting Scale, total", ""),
    }
    for cov, (var, cat) in mimic_rows.items():
        out = ROOT / "psychometrics/mimic" / cov / f"model_mimic_{cov.lower()}.out"
        m = re.search(r"Number of observations\s+(\d+)", out.read_text(encoding="utf-8", errors="ignore"))
        check(S, f"MIMIC {cov}: Table 1 n = Mplus N", int(m.group(1)), int(row(var, cat)["n_with_data"]),
              source=f"{src}; psychometrics/mimic/{cov}/{out.name}")
        if cov not in ("AntCor",):
            n_lab = f"MIMIC {cov}: covariate row carries the MIMIC name"
            check(S, n_lab, True, f"MIMIC covariate {cov} (" in str(row(var, cat)["note"]), source=src)
    for cat, dummy in (("partial course", "AntCor_P"), ("complete course", "AntCor_C")):
        check(S, f"MIMIC {dummy}: covariate row carries the MIMIC name", True,
              f"MIMIC covariate {dummy} (" in str(row("Antenatal corticosteroids", cat)["note"]), source=src)


def check_cohort() -> None:
    """Cohort counts and study dates of the dHCP release
    (results/metrics/cohort_counts.json, analysis/cohort_counts.py)."""
    S = "Cohort"
    c = json.loads((RES / "metrics/cohort_counts.json").read_text(encoding="utf-8"))
    src = "results/metrics/cohort_counts.json (analysis/cohort_counts.py)"
    check(S, "instruments in data/raw/dhcp_txt", 12, len(c["instruments"]), source=src)
    check(S, "distinct subject IDs across the instruments", 984, c["n_subjects_all_instruments"], source=src)
    check(S, "subjects with a bsid_iii01 record (18-month Bayley-III)", 739, c["n_subjects_bsid_iii01"], source=src)
    check(S, "infants with item-level cognition responses (N)", 739, c["n_infants_item_level_cognition"], source=src)
    check(S, "analysed infants with a bsid_iii01 record", 739, c["analysed_infants_with_bsid_record"], source=src)
    rp = c["recruitment_period"]["all_subjects"]
    check(S, "recruitment period (cpenr01 enrolment dates, all subjects)", "2014-03-06 to 2020-10-31",
          f"{rp['first']} to {rp['last']}", source=src)
    rp = c["recruitment_period"]["analysed_infants"]
    check(S, "recruitment period (analysed infants)", "2014-03-06 to 2020-10-31",
          f"{rp['first']} to {rp['last']}", source=src)
    bd = c["bayley_assessment_dates"]["all_subjects"]
    check(S, "18-month assessment dates (bsid_iii01, earliest to latest)", "2015-10-02 to 2022-10-24",
          f"{bd['first']} to {bd['last']}", source=src)
    check(S, "18-month assessments dated", 739, bd["n"], source=src)


def check_translation() -> None:
    S = "Objective 1 - translation"
    s1 = pd.read_csv(RES / "tables/table_s1_cascade_pairs.csv")
    src = "results/tables/table_s1_cascade_pairs.csv"
    sim = s1["cosine_similarity_fulltext"]
    check(S, "n pairs", 39, len(s1), source=src)
    check(S, "mean full-text similarity", 0.846, sim.mean(), 0.0005, src)
    check(S, "median full-text similarity", 0.902, sim.median(), 0.0005, src)
    check(S, "min similarity", 0.532, sim.min(), 0.0005, src); check(S, "max similarity", 0.983, sim.max(), 0.0005, src)
    st = s1["pairing_stage"].value_counts()
    check(S, "title-locked pairs (Fig 2A)", 27, st.get("title-locked", 0), source=src)
    check(S, "full-text fallback pairs (Fig 2A)", 12, st.get("full-text fallback", 0), source=src)

    a = pd.read_csv(RES / "tables/bayley3_item_domain_assignments.csv")
    src = "results/tables/bayley3_item_domain_assignments.csv"
    cnt = a["domain"].value_counts()
    for d, n in (("ATT", 22), ("WM", 19), ("GDPS", 23), ("FS", 20), ("HOP", 7)):
        check(S, f"{d} items (91-item taxonomy)", n, int(cnt.get(d, 0)), source=src)
    check(S, "unpaired items", 52, int((a["assignment_source"] != "cascade_pairing").sum()), source=src)
    check(S, "items flagged for expert review", 6, int(a["uc_flagged"].sum()), source=src)
    ext = a.loc[a["assignment_source"] != "cascade_pairing", "domain"].value_counts()
    for d, n in (("GDPS", 18), ("ATT", 13), ("WM", 7), ("HOP", 3)):
        check(S, f"{d} items without a Bayley-4 counterpart", n, int(ext.get(d, 0)), source=src)
    dh = a[a["in_dhcp_range"]]
    check(S, "items in the dHCP window (34-68)", 35, len(dh), source=src)
    check(S, "ATT items in window", 1, int((dh["domain"] == "ATT").sum()), source=src)
    check(S, "HOP items in window", 0, int((dh["domain"] == "HOP").sum()), source=src)
    check(S, "items retained in the CFA", 23, int(a["in_cfa_model"].sum()), source=src)

    pc = pd.read_csv(RES / "tables/fig2_panelC_theoretical.csv")
    src = "results/tables/fig2_panelC_theoretical.csv (figures/fig2_embedding.py --export-panel-c)"
    within = pc[pc["comparison"] == "within"].set_index("focus_factor")["mean_cosine"]
    check(S, "within-domain cohesion FS", 0.618, within["FS"], 0.0005, src)
    check(S, "within-domain cohesion HOP", 0.610, within["HOP"], 0.0005, src)
    check(S, "within-domain cohesion GDPS", 0.519, within["GDPS"], 0.0005, src)
    between_max = pc[pc["comparison"] == "between"].groupby("focus_factor")["mean_cosine"].max()
    check(S, "within > between for every domain", True, bool((within > between_max[within.index]).all()), source=src)


def check_psychometrics() -> None:
    S = "Objective 1 - psychometrics"
    fit = pd.read_csv(RES / "tables/table_s3c_cfa_fit_statistics.csv").set_index("model")
    src = "results/tables/table_s3c_cfa_fit_statistics.csv (Mplus .out)"
    f3, f1 = fit.iloc[0], fit.iloc[1]
    for k, v in (("rmsea", 0.060), ("cfi", 0.929), ("tli", 0.921), ("srmr", 0.108),
                 ("rmsea_ci_low", 0.055), ("rmsea_ci_high", 0.064)):
        check(S, f"3-factor {k}", v, f3[k], 0.0005, src)
    for k, v in (("rmsea", 0.075), ("cfi", 0.885), ("tli", 0.874), ("srmr", 0.113)):
        check(S, f"unidimensional {k}", v, f1[k], 0.0005, src)
    # nested-model difference test (Results: "fitted significantly worse ... Δχ²(3)=246·1, p<0·0001")
    check(S, "DIFFTEST unidimensional vs 3-factor chi2", 246.1, f1["difftest_vs_3factor_chi2"], 0.05, src)
    check(S, "DIFFTEST df", 3, f1["difftest_df"], source=src)
    check(S, "DIFFTEST p < .0001", True, bool(f1["difftest_p"] < 0.0001), source=src)
    lo = pd.read_csv(RES / "tables/table_s3a_cfa_3factor_loadings.csv")
    src = "results/tables/table_s3a_cfa_3factor_loadings.csv"
    check(S, "min standardised loading", 0.41, lo["std_loading"].min(), 0.005, src)
    check(S, "max standardised loading", 0.98, lo["std_loading"].max(), 0.005, src)
    check(S, "all loadings p < .001", True, bool((lo["p_value"] < 0.001).all()), source=src)
    fc = lo["factor"].value_counts()
    for f, n in (("WM", 4), ("GDPS", 8), ("FS", 11)):
        check(S, f"{f} indicators", n, int(fc.get(f, 0)), source=src)
    co = pd.read_csv(RES / "tables/table_s3b_cfa_3factor_correlations.csv")["std_correlation"]
    check(S, "min factor correlation (Discussion: 0.54)", 0.54, co.min(), 0.005, "results/tables/table_s3b_cfa_3factor_correlations.csv")
    check(S, "max factor correlation (Discussion: 0.70)", 0.70, co.max(), 0.005, "results/tables/table_s3b_cfa_3factor_correlations.csv")

    # --- MIMIC ---
    S = "Objective 1 - MIMIC"
    m = pd.read_csv(RES / "tables/table_s4a_mimic_covariate_associations.csv")
    src = "results/tables/table_s4a_mimic_covariate_associations.csv"

    def est(pred, factor, model="3F"):
        row = m[(m["predictor"] == pred) & (m["factor"] == factor) & (m["model"] == model)]
        return (r(row["estimate"].iloc[0]), r(row["p_value"].iloc[0])) if len(row) else (None, None)

    for pred, fac, b, p in (("GA", "FS", 0.159, None), ("HDCIRC", "FS", 0.179, None), ("BIRTHWT", "FS", 0.139, None),
                            ("ANTCOR_C", "FS", -0.276, 0.026), ("SEX", "GDPS", -0.365, None), ("SEX", "WM", -0.241, 0.009),
                            ("SEX", "FS", -0.094, 0.228), ("EPDSTOT", "GDPS", -0.156, 0.001), ("MOMAGE", "WM", 0.125, 0.009)):
        e, pv = est(pred, fac)
        check(S, f"{pred} -> {fac} beta", b, e, 0.0005, src)
        if p is not None:
            check(S, f"{pred} -> {fac} p", p, pv, 0.0005, src)
    e, pv = est("SEX", "COG (unidimensional)", "UNIDIM")
    check(S, "SEX -> composite beta", -0.172, e, 0.0005, src); check(S, "SEX -> composite p", 0.024, pv, 0.0005, src)
    e, pv = est("EPDSTOT", "COG (unidimensional)", "UNIDIM")
    check(S, "EPDS -> composite beta", -0.107, e, 0.0005, src); check(S, "EPDS -> composite p", 0.013, pv, 0.0005, src)
    check(S, "EPDS -> WM p", 0.052, est("EPDSTOT", "WM")[1], 0.0005, src)
    check(S, "EPDS -> FS p", 0.063, est("EPDSTOT", "FS")[1], 0.0005, src)
    for pred, lo_, hi_, lab in (("MOMENG", 0.198, 0.309, "maternal English"), ("STIMENV", 0.185, 0.242, "stimulating home")):
        sub = m[(m["predictor"] == pred) & (m["model"] == "3F")]
        check(S, f"{lab} min beta", lo_, sub["estimate"].min(), 0.0005, src)
        check(S, f"{lab} max beta", hi_, sub["estimate"].max(), 0.0005, src)
        check(S, f"{lab} all p <= .017", True, bool((sub["p_value"] <= 0.017).all()), source=src)
    ns = m[m["predictor"].isin(["PRLAX", "PROVR", "PRVRB", "PRSTY", "GDIAB", "IUGR", "PRECL"])]
    check(S, "parenting/GDiab/IUGR/pre-eclampsia: no p < .05", True,
          bool((ns["p_value"] >= 0.05).all()), source=src)
    # Paternal education (Results): GDPS beta = -0.135, p = .012 (STDYX);
    # composite p = .119; WM = GDPS DIFFTEST chi2(1) = 2.53, p = .11.
    # WM and FS are n.s., so GDPS is the only significant factor.
    dad = m[(m["predictor"] == "DADEDU") & (m["p_value"] < 0.05)]
    check(S, "paternal education: GDPS is the only significant factor",
          "GDPS", ",".join(dad["factor"]), source=src)
    for fac, b, p in (("WM", -0.042, 0.457), ("GDPS", -0.135, 0.012), ("FS", -0.046, 0.269)):
        e, pv = est("DADEDU", fac)
        check(S, f"paternal education -> {fac} beta", b, e, 0.0005, src)
        check(S, f"paternal education -> {fac} p", p, pv, 0.0005, src)
    e, pv = est("DADEDU", "COG (unidimensional)", "UNIDIM")
    check(S, "paternal education -> composite beta", -0.065, e, 0.0005, src)
    check(S, "paternal education -> composite p", 0.119, pv, 0.0005, src)
    d = pd.read_csv(RES / "tables/table_s4b_mimic_difftest.csv").set_index("constraint")
    src = "results/tables/table_s4b_mimic_difftest.csv"
    for c, chi, p in (("GA: WM = FS", 8.81, 0.003), ("HdCirc: WM = FS", 8.41, 0.004), ("BirthWt: WM = FS", 7.61, 0.006),
                      ("AntCor_C: WM = FS", 6.80, 0.009), ("MomAge: WM = GDPS", 5.98, 0.015),
                      ("DadEdu: WM = GDPS", 2.53, 0.112)):
        check(S, f"DIFFTEST {c} chi2", chi, d.loc[c, "chi2_diff"], 0.005, src)
        check(S, f"DIFFTEST {c} p", p, d.loc[c, "p_value"], 0.0005, src)
    check(S, "DIFFTEST DadEdu WM = GDPS not significant", False, bool(d.loc["DadEdu: WM = GDPS", "p_value"] < 0.05), source=src)

    # --- outcomes ---
    S = "Objective 1 - outcomes"
    o = pd.read_csv(RES / "tables/table_s5_factor_outcome_correlations.csv")
    src = "results/tables/table_s5_factor_outcome_correlations.csv"

    def sub(code, model="3F"):
        return o[(o["outcome_code"] == code) & (o["model"] == model)]

    check(S, "language min r", 0.45, sub("BayLan")["estimate"].min(), 0.005, src)
    check(S, "language max r", 0.64, sub("BayLan")["estimate"].max(), 0.005, src)
    check(S, "motor min r", 0.39, sub("BayMot")["estimate"].min(), 0.005, src)
    check(S, "motor max r", 0.44, sub("BayMot")["estimate"].max(), 0.005, src)
    check(S, "Q-CHAT min r", -0.39, sub("QchatTot")["estimate"].min(), 0.005, src)
    check(S, "Q-CHAT max r", -0.28, sub("QchatTot")["estimate"].max(), 0.005, src)
    ci = sub("CBInt").set_index("factor")
    check(S, "CBCL internalising GDPS r", -0.153, ci.loc["GDPS", "estimate"], 0.0005, src)
    check(S, "CBCL internalising GDPS p < .001", True, bool(ci.loc["GDPS", "p_value"] < 0.001), source=src)
    check(S, "CBCL internalising WM r", -0.119, ci.loc["WM", "estimate"], 0.0005, src)
    check(S, "CBCL internalising WM p", 0.011, ci.loc["WM", "p_value"], 0.0005, src)
    check(S, "CBCL internalising FS r", -0.031, ci.loc["FS", "estimate"], 0.0005, src)
    cu = sub("CBInt", "UNIDIM").iloc[0]
    check(S, "CBCL internalising composite r", -0.075, cu["estimate"], 0.0005, src)
    check(S, "CBCL internalising composite p", 0.066, cu["p_value"], 0.0005, src)
    ct = sub("CBTot").set_index("factor")
    check(S, "CBCL total GDPS r", -0.102, ct.loc["GDPS", "estimate"], 0.0005, src)
    check(S, "CBCL total GDPS p", 0.021, ct.loc["GDPS", "p_value"], 0.0005, src)
    cut = sub("CBTot", "UNIDIM").iloc[0]
    check(S, "CBCL total composite r", -0.037, cut["estimate"], 0.0005, src)
    check(S, "CBCL total composite p", 0.366, cut["p_value"], 0.0005, src)
    check(S, "CBCL externalising: no p < .05", True, bool((sub("CBExt")["p_value"] >= 0.05).all()), source=src)


def check_concordance() -> None:
    import _validation as val
    m = json.loads((RES / "metrics/manuscript_metrics.json").read_text())
    src = "results/metrics/manuscript_metrics.json (analysis/concordance/*.py)"
    S = "Objective 2 - Bayley-4"
    check(S, "ARI vs expert-derived", 0.49, m["Bayley4_Sankey_Theo_vs_Det"]["ARI"], 0.005, src)
    check(S, "NMI vs expert-derived", 0.63, m["Bayley4_Sankey_Theo_vs_Det"]["NMI"], 0.005, src)
    il0 = m["Bayley4_ItemLevel_Theo_vs_Det"]
    check(S, "item agreement fraction (39 expert-assigned items)", "24/39", il0["overall_agreement_fraction"], source=src)
    check(S, "item agreement %", 61.5, il0["overall_agreement_rate_pct"], 0.05, src)
    check(S, "item agreement over all 40 clustered items (not in the text)", "24/40",
          il0["overall_agreement_fraction_all_clustered"], source=src)
    check(S, "item agreement % over all 40 clustered items (not in the text)", 60.0,
          il0["overall_agreement_rate_pct_all_clustered"], 0.05, src)
    check(S, "ARI vs empirical", 0.42, m["Bayley4_Sankey_Det_vs_Emp"]["ARI"], 0.005, src)
    check(S, "NMI vs empirical", 0.56, m["Bayley4_Sankey_Det_vs_Emp"]["NMI"], 0.005, src)
    check(S, "ARI expert-derived vs empirical", 0.19, m["Bayley4_Sankey_Theo_vs_Emp"]["ARI"], 0.005, src)
    check(S, "NMI expert-derived vs empirical", 0.40, m["Bayley4_Sankey_Theo_vs_Emp"]["NMI"], 0.005, src)
    M4, *_ = val.load_b4(ROOT)
    i = {f: k for k, f in enumerate(val.ORDER)}
    src2 = "figures/_validation.py confusion matrix (Figure 5A)"
    check(S, "ATT recovered 9/9", "9/9", f"{M4[i['ATT'], i['ATT']]}/{M4[i['ATT']].sum()}", source=src2)
    check(S, "FS recovered 9/10", "9/10", f"{M4[i['FS'], i['FS']]}/{M4[i['FS']].sum()}", source=src2)

    # --- item counts: 39 items carry an expert domain in Aylward's table (42
    # memberships; COG_035 GDPS+FS, COG_048 GDPS+HOP, COG_051 FS+HOP); the
    # clustering ran on 40 items because COG_047 appears only in the empirical
    # model.  The text says "40 Bayley-4 items" for the clustering (Methods,
    # Figure 1B legend), "39" for the expert assignment, the cascade pairs and
    # the recovery denominator (24/39).
    import yaml
    ref = yaml.safe_load((ROOT / "config/reference_models_b4.yaml").read_text(encoding="utf-8"))
    src = "config/reference_models_b4.yaml"
    memb = {f: len(v) for f, v in ref["theoretical"].items() if isinstance(v, list)}
    uniq = sorted({it for v in ref["theoretical"].values() if isinstance(v, list) for it in v})
    multi = sorted(it for it in uniq if sum(it in v for v in ref["theoretical"].values() if isinstance(v, list)) > 1)
    check(S, "Aylward facet memberships (ATT/WM/GDPS/FS/HOP)", "9/12/5/10/6",
          "/".join(str(memb[f]) for f in ("attention", "working_memory", "goal_directed_problem_solving",
                                          "flexibility_shift", "higher_order_processing")), source=src)
    check(S, "Bayley-4 items with an expert domain", 39, len(uniq), source=src)
    check(S, "multi-factor items", "COG_035,COG_048,COG_051", ",".join(multi), source=src)
    clus = pd.read_csv(ROOT / "data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
                       / "03_clustering/cluster_assignments.csv")
    src = "data/outputs/step2_b4_validation/.../03_clustering/cluster_assignments.csv"
    check(S, "Bayley-4 items clustered (Methods, Figure 1B legend: 40)", 40, len(clus), source=src)
    check(S, "clustered item without an expert domain", "COG_047", ",".join(sorted(set(clus["item_id"]) - set(uniq))), source=src)
    il = m["Bayley4_ItemLevel_Theo_vs_Det"]
    src = "results/metrics/manuscript_metrics.json (analysis/concordance/b4_item_level_analysis.py)"
    check(S, "items with expert domain (item-level analysis)", 39, il["n_items_with_expert_domain"], source=src)
    check(S, "multi-factor items (item-level analysis)", 3, il["n_items_multi_factor"], source=src)
    check(S, "concordance under the primary-domain rule (39 items)", "22/39", il["primary_domain_agreement_fraction"], source=src)
    # per-domain recovery: any-factor rule, multi-factor items credited to every listed domain
    da = il["domain_specific_accuracy"]
    for dom, frac in (("ATT", "9/9"), ("WM", "5/12"), ("GDPS", "0/5"), ("FS", "9/10"), ("HOP", "1/6")):
        check(S, f"{dom} recovered (any-factor rule)", frac, f"{da[dom]['n_agree']}/{da[dom]['n_items']}", source=src)
    for dom, frac in (("ATT", "9/9"), ("WM", "5/12"), ("GDPS", "0/3"), ("FS", "7/8"), ("HOP", "0/4")):
        check(S, f"{dom} recovered (single-domain items only)", frac,
              f"{da[dom]['n_agree_single_domain']}/{da[dom]['n_items_single_domain']}", source=src)
    # cascade pairs: the 39 expert-assigned items, multi-factor items resolved first-wins
    s1 = pd.read_csv(RES / "tables/table_s1_cascade_pairs.csv")
    src = "results/tables/table_s1_cascade_pairs.csv"
    check(S, "paired Bayley-4 items = the 39 with an expert domain", True, sorted(s1["b4_item_id"]) == uniq, source=src)

    # --- discordant items and the expert review.  Any-factor rule: 40 - 24 = 16
    # discordant items (15 with a different expert domain + COG_047 without
    # one), as the text reports.  The workbook held 17 rows because COG_048
    # was flagged by a label-string mismatch; neither expert rated it, so every
    # kappa is computed on the 16 discordant items (Table S8a).
    check(S, "discordant items (any-factor rule, 40 items)", 16, il["n_discordant"], source=src)
    k4 = json.loads((ROOT / "expert_review/records/coauthor_review_b4_kappa.json").read_text())
    src = "expert_review/records/coauthor_review_b4_kappa.json"
    check(S, "review workbook rows (17; the text reports the 16 discordant)", 17, k4["n_items"], source=src)
    check(S, "discordant items in the workbook (any-factor rule)", 16, k4["n_discordant_any_factor"], source=src)
    check(S, "items rated by at least one expert", 16, k4["n_rated_items"], source=src)
    check(S, "unrated workbook row", "COG_048", ",".join(k4["items_unrated"]), source=src)
    check(S, "reviewed set = item-level discordant set + COG_048", True,
          sorted(k4["items_reviewed"]) == sorted(il["discordant_items"] + ["COG_048"]), source=src)
    check(S, "kappa expert 1 vs 2", 0.32, k4["kappa"]["Expert 1 vs Expert 2 (IRR)"], 0.005, src)
    check(S, "kappa expert 1 vs Aylward", 0.34, k4["kappa"]["Expert 1 vs Aylward (reference)"], 0.005, src)
    check(S, "kappa expert 2 vs Aylward", 0.42, k4["kappa"]["Expert 2 vs Aylward (reference)"], 0.005, src)
    check(S, "kappa consensus vs clusters", -0.33, k4["kappa"]["Consensus vs Detected (NLP)"], 0.005, src)
    check(S, "items with expert consensus", 8, k4["n_consensus"], source=src)

    # --- pair-level intermediacy (Discussion: "between the two rather than outside both")
    S = "Objective 2 - pair-level intermediacy"
    pl = m["Bayley4_PairLevel_Intermediacy"]
    src = "results/metrics/manuscript_metrics.json (analysis/concordance/b4_pair_level_intermediacy.py)"
    st = pl["strict"]
    check(S, "items (single-domain, in both references)", 31, st["n_items"], source=src)
    check(S, "consensus-together pairs kept together", "30/39",
          f"{st['consensus_together']['kept_together']}/{st['consensus_together']['n']}", source=src)
    check(S, "consensus-apart pairs kept apart", "287/299",
          f"{st['consensus_apart']['kept_apart']}/{st['consensus_apart']['n']}", source=src)
    check(S, "consensus-apart kept apart % (Results: 96%)", 96.0, st["consensus_apart"]["pct"], 0.05, src)
    check(S, "disputed pairs resolved toward expert / empirical (Table S7 only)", "67/60",
          f"{st['disputed']['resolved_toward_expert']}/{st['disputed']['resolved_toward_empirical']}", source=src)
    tp = st["together_pairs"]
    check(S, "detected merges fewer pairs than either reference (not a coarser partition)", True,
          tp["detected"] < min(tp["expert"], tp["empirical"]), source=src)
    check(S, "together-pairs expert/empirical/detected", "108/97/87", f"{tp['expert']}/{tp['empirical']}/{tp['detected']}", source=src)
    af = pl["any_factor"]
    check(S, "any-factor treatment: consensus-together kept", "40/52",
          f"{af['consensus_together']['kept_together']}/{af['consensus_together']['n']}", source=src)
    check(S, "any-factor treatment: consensus-apart kept apart", "345/358",
          f"{af['consensus_apart']['kept_apart']}/{af['consensus_apart']['n']}", source=src)
    check(S, "any-factor treatment: disputed toward expert / empirical", "81/70",
          f"{af['disputed']['resolved_toward_expert']}/{af['disputed']['resolved_toward_empirical']}", source=src)

    S = "Objective 3 - Bayley-III"
    src = "results/metrics/manuscript_metrics.json (analysis/concordance/b3_*.py)"
    check(S, "ARI vs translated", 0.39, m["Bayley3_Sankey_Theo_vs_Det"]["ARI"], 0.005, src)
    check(S, "NMI vs translated", 0.48, m["Bayley3_Sankey_Theo_vs_Det"]["NMI"], 0.005, src)
    check(S, "item agreement fraction", "31/91", m["Bayley3_ItemLevel_Theo_vs_Det"]["overall_agreement_fraction"], source=src)
    check(S, "item agreement %", 34.1, m["Bayley3_ItemLevel_Theo_vs_Det"]["overall_agreement_rate_pct"], 0.05, src)
    M3, *_ = val.load_b3(ROOT)
    src2 = "figures/_validation.py confusion matrix (Figure 5B)"
    check(S, "ATT recovered 18/22", "18/22", f"{M3[i['ATT'], i['ATT']]}/{M3[i['ATT']].sum()}", source=src2)
    check(S, "HOP items joining WM 5/7", "5/7", f"{M3[i['HOP'], i['WM']]}/{M3[i['HOP']].sum()}", source=src2)
    # --- expert review.  The 61 reviewed items are the 55-item discordant sheet
    # plus the 6 UC-flagged items (4 of them also discordant): 59 items were
    # discordant under the pre-review algorithmic structure; the experts'
    # overrides made COG_033 discordant (60 under the final structure) and
    # COG_027 (UC-flagged, concordant) was reviewed but was never discordant.
    # The text reports the 60-item set (Table S8b): kappa 0.32 between experts,
    # 0.27 / 0.19 vs the pre-override algorithmic labels, -0.12 / -0.22 vs the
    # bottom-up clusters; 0.35 / 0.25 vs the final structure (Table S8 footnote).
    k3 = json.loads((ROOT / "expert_review/records/coauthor_review_b3_kappa.json").read_text())
    src = "expert_review/records/coauthor_review_b3_kappa.json"
    check(S, "items reviewed (61; the text reports the 60 discordant)", 61, k3["n_reviewed"], source=src)
    check(S, "discordant sheet rows", 55, k3["n_discordant_sheet"], source=src)
    check(S, "UC-flagged items in the review", 6, k3["n_uc_flagged"], source=src)
    check(S, "kappa AD vs AH", 0.33, k3["kappa"]["AD vs AH (IRR)"], 0.005, src)
    check(S, "kappa AD vs translated", 0.28, k3["kappa"]["AD vs Step 1 (theoretical)"], 0.005, src)
    check(S, "kappa AH vs translated", 0.20, k3["kappa"]["AH vs Step 1 (theoretical)"], 0.005, src)
    check(S, "kappa AD vs bottom-up", -0.11, k3["kappa"]["AD vs Step 3 (NLP)"], 0.005, src)
    check(S, "kappa AH vs bottom-up", -0.20, k3["kappa"]["AH vs Step 3 (NLP)"], 0.005, src)
    fd = k3["final_discordant_subset"]
    check(S, "discordant under the final structure (ledger)", 60, fd["n_items"], source=src)
    check(S, "reviewed but concordant", "COG_027", ",".join(fd["reviewed_but_concordant"]), source=src)
    led = pd.read_csv(RES / "tables/mismatches_ledger_b3.csv")
    check(S, "mismatches ledger rows", 60, len(led), source="results/tables/mismatches_ledger_b3.csv")
    check(S, "ledger = reviewed set minus COG_027", True,
          sorted(led["item_id"]) == sorted(set(k3["items_reviewed"]) - {"COG_027"}), source=src)
    a = pd.read_csv(RES / "tables/bayley3_item_domain_assignments.csv").set_index("item_id")
    det3 = M3_det(ROOT)
    src_a = "results/tables/bayley3_item_domain_assignments.csv"
    ov = sorted(a.index[a["assignment_source"] == "expert_review_override"])
    check(S, "expert overrides", "COG_033,COG_065,COG_069", ",".join(ov), source=src_a)
    pre = sorted(i for i in a.index if a.loc[i, "algorithmic_domain"] != det3[i])
    post = sorted(i for i in a.index if a.loc[i, "domain"] != det3[i])
    check(S, "discordant before the overrides (algorithmic vs bottom-up)", 59, len(pre), source=src_a)
    check(S, "discordant after the overrides (final vs bottom-up)", 60, len(post), source=src_a)
    check(S, "item made discordant by its override", "COG_033", ",".join(sorted(set(post) - set(pre))), source=src_a)
    check(S, "items made concordant by the overrides", "", ",".join(sorted(set(pre) - set(post))), source=src_a)
    for lab, key, v in (("AD vs AH (text)", "AD vs AH (IRR)", 0.32),
                        ("AD vs translated, pre-override (text)", "AD vs Step 1 (theoretical)", 0.27),
                        ("AH vs translated, pre-override (text)", "AH vs Step 1 (theoretical)", 0.19),
                        ("AD vs final translated", "AD vs final translated", 0.35), ("AH vs final translated", "AH vs final translated", 0.25),
                        ("AD vs bottom-up (text)", "AD vs Step 3 (NLP)", -0.12), ("AH vs bottom-up (text)", "AH vs Step 3 (NLP)", -0.22)):
        check(S, f"60-item set: kappa {lab}", v, fd["kappa"][key], 0.005, src)


def check_supplementary_tables() -> None:
    """Tables S6-S8 (results/tables/) carry the numbers the text cites them for."""
    S = "Supplementary tables S6-S8"
    s6 = pd.read_csv(RES / "tables/table_s6_b4_domain_recovery.csv").set_index("domain")
    src = "results/tables/table_s6_b4_domain_recovery.csv"
    check(S, "S6 all items, any-factor rule", "24/39",
          f"{s6.loc['All items', 'recovered_any_factor']}/{s6.loc['All items', 'items_any_factor']}", source=src)
    check(S, "S6 all items %", 61.5, s6.loc["All items", "pct_any_factor"], 0.05, src)
    for dom, frac in (("ATT", "9/9"), ("WM", "5/12"), ("GDPS", "0/5"), ("FS", "9/10"), ("HOP", "1/6")):
        check(S, f"S6 {dom} any-factor", frac, f"{s6.loc[dom, 'recovered_any_factor']}/{s6.loc[dom, 'items_any_factor']}", source=src)
    for dom, frac in (("GDPS", "0/3"), ("FS", "7/8"), ("HOP", "0/4")):
        check(S, f"S6 {dom} single-domain", frac,
              f"{s6.loc[dom, 'recovered_single_domain']}/{s6.loc[dom, 'items_single_domain']}", source=src)
    s7 = pd.read_csv(RES / "tables/table_s7_b4_pair_level_intermediacy.csv")
    src = "results/tables/table_s7_b4_pair_level_intermediacy.csv"
    s7 = s7.set_index(["pair_class", "quantity"])["single_domain_31_items"]
    check(S, "S7 consensus-apart pairs (31 items)", 299, int(s7[("Consensus-apart pairs", "Pairs")]), source=src)
    check(S, "S7 consensus-apart kept apart (Results: 96%)", "287 (96%)",
          s7[("Consensus-apart pairs", "Kept apart by the clusters")], source=src)
    check(S, "S7 consensus-together kept together", "30 (77%)",
          s7[("Consensus-together pairs", "Kept together by the clusters")], source=src)
    check(S, "S7 disputed pairs expert / empirical", "67/60",
          f"{s7[('Disputed pairs', 'Resolved toward the expert model')]}/{s7[('Disputed pairs', 'Resolved toward the empirical model')]}",
          source=src)
    a = pd.read_csv(RES / "tables/table_s8a_expert_review_b4.csv")
    src = "results/tables/table_s8a_expert_review_b4.csv"
    check(S, "S8a discordant Bayley-4 items", 16, len(a), source=src)
    check(S, "S8a items with a consensus", 8, int((a["consensus"] != "\u2014").sum()), source=src)
    check(S, "S8a = mismatches ledger (Bayley-4)", True,
          sorted(a["item_id"]) == sorted(pd.read_csv(RES / "tables/mismatches_ledger_b4.csv")["item_id"]), source=src)
    b = pd.read_csv(RES / "tables/table_s8b_expert_review_b3.csv")
    src = "results/tables/table_s8b_expert_review_b3.csv"
    check(S, "S8b discordant Bayley-III items", 60, len(b), source=src)
    check(S, "S8b items with a consensus", 24, int((b["consensus"] != "\u2014").sum()), source=src)
    check(S, "S8b expert overrides (algorithmic != final)", "COG_033,COG_065,COG_069",
          ",".join(sorted(b.loc[b["algorithmic_label"] != b["final_label"], "item_id"])), source=src)
    check(S, "S8b = mismatches ledger (Bayley-III)", True,
          sorted(b["item_id"]) == sorted(pd.read_csv(RES / "tables/mismatches_ledger_b3.csv")["item_id"]), source=src)


# ---------------------------------------------------------------------------
# Optional: the wording of the manuscript itself
# ---------------------------------------------------------------------------
def _docx_text(path: Path) -> str:
    try:
        import docx  # python-docx
        from docx.oxml.ns import qn
        d = docx.Document(str(path))
        # walk every paragraph in the body, including those inside content
        # controls (w:sdt), tables and text boxes, which doc.paragraphs skips
        parts = ["".join(t.text or "" for t in p.iter(qn("w:t"))) for p in d.element.body.iter(qn("w:p"))]
        text = "\n".join(parts)
    except ImportError:
        import zipfile
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
        text = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
    return re.sub(r"[ \t\u00a0]+", " ", text)


def resolve_manuscript(arg: str | None) -> Path | None:
    """--manuscript PATH (file or folder); default = the git-ignored manuscript/ folder."""
    cand = Path(arg).expanduser() if arg else MANUSCRIPT_DIR
    if cand.is_dir():
        files = sorted(cand.glob("*.docx"), key=lambda f: f.stat().st_mtime)
        return files[-1] if files else None
    return cand if cand.is_file() else None


def check_manuscript_text(path: Path) -> None:
    text = _docx_text(path)
    S = "Manuscript wording"
    src = str(path)
    present = [
        ("39 pairs", "39 Bayley-4 to Bayley-III pairs"),
        ("40 items clustered (Methods)", "embeddings of the 40 Bayley-4 items"),
        ("40 items clustered (Figure 1B legend)", "the 39 expert-assigned items plus one assigned only in the empirical model"),
        ("recovery 24/39, 61.5 %", "24/39 items, 61\u00b75%"),
        ("ATT 9/9, FS 9/10", "ATT (9/9) and FS (9/10)"),
        ("16 discordant Bayley-4 items, kappa 0.32", "16 discordant items (\u03ba=0\u00b732 between experts)"),
        ("kappa vs Aylward 0.34, 0.42", "(\u03ba=0\u00b734, 0\u00b742)"),
        ("kappa -0.33 on eight items", "\u03ba=\u22120\u00b733 on the eight items"),
        ("60 discordant Bayley-III items, kappa 0.32", "60 discordant items (\u03ba=0\u00b732 between experts)"),
        ("kappa vs translated 0.27, 0.19", "(\u03ba=0\u00b727, 0\u00b719)"),
        ("kappa vs bottom-up -0.12, -0.22", "(\u03ba=\u22120\u00b712, \u22120\u00b722)"),
        ("paternal education sentence", "paternal education (\u03b2=\u22120\u00b7135, p=0\u00b7012; composite p=0\u00b7119; "
                                        "WM\u2013GDPS \u0394\u03c7\u00b2(1)=2\u00b753, p=0\u00b711)"),
        ("96 % consensus-apart kept apart", "kept apart 96% of the item pairs"),
        ("NICU 166 (22.5 %)", "166 infants (22\u00b75%)"),
        ("birth-weight SD 0.87", "SD=0\u00b787"),
        ("nested-model test", "\u0394\u03c7\u00b2(3)=246\u00b71, p<0\u00b70001"),
        ("expert-review ledger cited as Table S8", "Table S8"),
    ]
    for label, needle in present:
        check(S, f"text contains: {label}", True, needle in text, source=src)
    check(S, "p values carry the leading zero (no p=\u00b7 / p<\u00b7 left)", 0,
          len(re.findall(r"p\s?[=<>\u2264\u2265]\s?\u00b7\d", text)), source=src)
    for label, needle in (("superseded 24/40", "24/40"), ("superseded 60.0 %", "60\u00b70%"),
                          ("superseded 17 discordant", "17 discordant"), ("superseded 61 discordant", "61 discordant")):
        check(S, f"text no longer contains: {label}", False, needle in text, source=src)


def M3_det(root: Path) -> dict:
    """item -> bottom-up (Step 3) short domain code, for the Bayley-III checks."""
    import _validation as val
    run = root / "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"
    clusters = pd.read_csv(run / "03_clustering/cluster_assignments.csv")
    labels = pd.read_csv(run / "04_labeling/cluster_concept_labels.csv")
    cid2short = {int(r["cluster_id"]): val._short(r["top_concept"]) for _, r in labels.iterrows()}
    return {r["item_id"]: cid2short.get(int(r["cluster"])) for _, r in clusters.iterrows()}


def run(manuscript: Path | None = None) -> list[dict]:
    CHECKS.clear()
    check_sample()
    check_table1()
    check_cohort()
    check_translation()
    check_psychometrics()
    check_concordance()
    check_supplementary_tables()
    if manuscript is not None:
        check_manuscript_text(manuscript)
    return CHECKS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manuscript", metavar="PATH",
                    help="manuscript .docx (or a folder holding one) for the wording checks; "
                         f"default: {MANUSCRIPT_DIR.relative_to(ROOT)}/ (git-ignored)")
    args = ap.parse_args(argv)
    manuscript = resolve_manuscript(args.manuscript)
    if manuscript is None:
        where = args.manuscript or f"{MANUSCRIPT_DIR.relative_to(ROOT)}/"
        print(f"[manuscript wording checks skipped: no .docx at {where}; pass --manuscript PATH]")
    else:
        print(f"[manuscript wording checks: {manuscript}]")
    checks = run(manuscript)
    width = max(len(c["label"]) for c in checks) + 2
    section = None
    for c in checks:
        if c["section"] != section:
            section = c["section"]
            print(f"\n== {section}")
        flag = "ok  " if c["ok"] else "FAIL"
        print(f"  [{flag}] {c['label']:<{width}} expected {c['expected']!s:>10}   observed {c['observed']!s}")
    n_fail = sum(not c["ok"] for c in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} manuscript numbers reproduced" + (f"; {n_fail} mismatch(es)" if n_fail else ""))
    out = RES / "metrics/manuscript_numbers_check.json"
    out.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    print(f"Written to {out.relative_to(ROOT)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
