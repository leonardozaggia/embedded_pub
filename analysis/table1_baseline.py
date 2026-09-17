#!/usr/bin/env python
"""
Table 1: baseline characteristics of the dHCP analysis sample (N = 739).

Writes results/tables/table1_baseline.csv, one row per table line, with the
columns

    block, variable, category, statistic, value, sd_or_pct, low, high,
    n_with_data, n_missing, note

  statistic = mean_sd     value = mean, sd_or_pct = SD, low/high = min/max
  statistic = n_pct       value = n,    sd_or_pct = % of n_with_data
  statistic = median_iqr  value = median, low/high = 25th/75th centile (unused)

`n_with_data` is the denominator actually used for the row; `n_missing` is
739 minus that.  Numbers are kept to three decimals (no journal rounding);
the birth-weight row, which the text reports to two decimals, keeps four so
that formatting it does not double-round the SD (0.8747, not 0.875 -> 0.88).

Sources (restricted dHCP / NDA instruments, docs/DATA_ACCESS.md):

  lpb01        sex, gestational age (scan-derived fallback from
               data/processed/cogn_id_GA.xlsx), birth weight, head
               circumference, NICU admission, suspected IUGR, gestational
               diabetes, hypertensive disorders, antenatal corticosteroids
  cpenr01      plurality, maternal age, parental age at leaving education,
               maternal English as first language
  bsid_iii01   age at assessment, Bayley-III composites
  epds01, pqmf01, stps01   EPDS, Parenting Scale, Cognitively Stimulating
               Parenting Scale (earliest visit, as in prepare_data.R)

Every covariate of the MIMIC models (Figure 4A, Table S4a) is coded exactly
as in psychometrics/prepare_data.R, so its Table 1 row has the same
definition and the same n as the column of psychometrics/combined.dat.  The
script re-derives each of these covariates from the instruments and compares
it with the corresponding combined.dat column (n, and value by value); the
comparison is printed and a mismatch stops the script.  Where the MIMIC
covariate is a transformed or dichotomised version of a raw variable, the raw
variable is reported and the transformation is stated in `note`.

The Methods "Sample" numbers already frozen in
results/metrics/sample_descriptives.json (analysis/sample_descriptives.py)
are re-derived here from the same helpers (analysis/growth_restriction.py,
analysis/age_at_assessment.py); the script prints a comparison with the
manuscript's Participants paragraph.

Usage (from the repository root):
    python analysis/table1_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import growth_restriction as gr      # noqa: E402  (sample, INTERGROWTH-21st centiles)
import age_at_assessment as aa       # noqa: E402  (corrected age)

DHCP = ROOT / "data/raw/dhcp_txt"
PSY = ROOT / "psychometrics"
OUT = ROOT / "results/tables/table1_baseline.csv"

COLUMNS = ["block", "variable", "category", "statistic", "value", "sd_or_pct",
           "low", "high", "n_with_data", "n_missing", "note"]
HC_SD_CLIP = 4          # prepare_data.R: head circumference beyond +/- 4 SD -> missing
ND = 3                  # decimals kept in the CSV
FOUR_DECIMALS = {"Birth weight, kg"}   # rows formatted to 2 decimals in the text: one extra decimal avoids double rounding


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_ndar_earliest(name: str) -> pd.DataFrame:
    """One row per subject, earliest interview_age first (prepare_data.R::dedup_ndar)."""
    df = gr.read_ndar(DHCP / name).copy()
    for c in df.columns:
        if c != "src_subject_id":
            df[c] = gr.to_num(df[c])
    df = df.sort_values(["src_subject_id", "interview_age"], kind="stable")
    return df.drop_duplicates("src_subject_id", keep="first")


def fl(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


class Table:
    def __init__(self, n_total: int):
        self.n = n_total
        self.rows: list[dict] = []

    def mean_sd(self, block, variable, x: pd.Series, note="", category=""):
        x = pd.to_numeric(x, errors="coerce").dropna()
        self.rows.append({"block": block, "variable": variable, "category": category,
                          "statistic": "mean_sd", "value": fl(x.mean()), "sd_or_pct": fl(x.std(ddof=1)),
                          "low": fl(x.min()), "high": fl(x.max()),
                          "n_with_data": int(x.size), "n_missing": self.n - int(x.size), "note": note})

    def n_pct(self, block, variable, category, count: int, denom: int, note=""):
        self.rows.append({"block": block, "variable": variable, "category": category,
                          "statistic": "n_pct", "value": int(count), "sd_or_pct": 100 * count / denom,
                          "low": None, "high": None,
                          "n_with_data": int(denom), "n_missing": self.n - int(denom), "note": note})

    def frame(self, rounded: bool = False) -> pd.DataFrame:
        """Full precision by default; `rounded` = the CSV layout (ND decimals,
        counts written as integers)."""
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        if not rounded:
            return df
        out = df.copy()
        # the manuscript formats birth weight to 2 decimals; three stored decimals
        # would double-round the SD (0.8747 -> 0.875 -> 0.88), so that row keeps 4
        nd = np.where(out["variable"].isin(FOUR_DECIMALS), ND + 1, ND)
        for c in ("sd_or_pct", "low", "high"):
            out[c] = [None if pd.isna(v) else round(float(v), int(d)) for v, d in zip(out[c], nd)]
        out["value"] = pd.Series([int(v) if st == "n_pct" else round(float(v), int(d))
                                  for v, st, d in zip(out["value"], out["statistic"], nd)],
                                 index=out.index, dtype=object)
        return out


# ---------------------------------------------------------------------------
# the sample, coded as in prepare_data.R
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    df = gr.build_sample()                      # 739 infants, cogn_id_GA.xlsx order
    lpb = gr.read_ndar(DHCP / "lpb01.txt")[
        ["src_subject_id", "baby_babyhc", "study_gestational_diabetes",
         "study_preeclampsia", "study_corticosteroids"]].copy()
    for c in lpb.columns[1:]:
        lpb[c] = gr.to_num(lpb[c])
    df = df.merge(lpb.drop_duplicates("src_subject_id"), on="src_subject_id", how="left")

    cpen = gr.read_ndar(DHCP / "cpenr01.txt")[
        ["src_subject_id", "pregnancy_size", "mother_age1", "mother_education1",
         "father_education1", "mother_eng_first_lan"]].copy()
    for c in cpen.columns[1:]:
        cpen[c] = gr.to_num(cpen[c])
    df = df.merge(cpen.drop_duplicates("src_subject_id"), on="src_subject_id", how="left")

    epds = read_ndar_earliest("epds01.txt")[["src_subject_id", "epds_tot"]]
    stps = read_ndar_earliest("stps01.txt")[["src_subject_id", "total_sps_home_environment"]]
    pqmf = read_ndar_earliest("pqmf01.txt")[["src_subject_id", "pm_laxness", "pm_overactivity",
                                              "pm_verbosity", "pm_parent_style"]]
    for extra in (epds, stps, pqmf):
        df = df.merge(extra, on="src_subject_id", how="left")

    ages = aa.build()[["src_subject_id", "chron_age", "corr_age", "bsid_cog_composite",
                       "bsid_lang_composite", "bsid_mot_composite"]]
    df = df.merge(ages, on="src_subject_id", how="left")

    # --- growth restriction (INTERGROWTH-21st, sex-specific) ------------------
    cent, _ = gr.bw_centile(df["BirthWt"], df["gest_days"], df["male"], gr.load_coeffs())
    df["bw_centile"] = cent * 100

    # --- MIMIC covariates, exactly as exported by prepare_data.R ---------------
    hc = df["baby_babyhc"]
    mu, sd = hc.mean(), hc.std(ddof=1)
    df["HdCirc_raw"] = hc
    df["HdCirc"] = hc.mask((hc - mu).abs() > HC_SD_CLIP * sd)
    sex = np.where(df["baby_gender"] == 1, 1.0, np.where(df["baby_gender"] == 2, 0.0, np.nan))
    df["Sex"] = sex                                     # 1 = male, 0 = female, no fallback
    df["GDiab"] = df["study_gestational_diabetes"]
    pe = df["study_preeclampsia"]
    df["PrEcl"] = np.where(pe == 0, 0.0, np.where(pe.isin([1, 2, 3]), 1.0, np.nan))
    ac = df["study_corticosteroids"]
    df["AntCor_P"] = np.where(ac.notna(), (ac == 1).astype(float), np.nan)
    df["AntCor_C"] = np.where(ac.notna(), (ac == 2).astype(float), np.nan)
    df["MomAge"] = df["mother_age1"]
    df["MomEdu"] = df["mother_education1"]
    df["DadEdu"] = df["father_education1"]
    df["MomEng"] = df["mother_eng_first_lan"]
    df["EPDSTot"] = df["epds_tot"]
    df["StimEnv"] = df["total_sps_home_environment"]
    df["PrLax"], df["PrOvr"] = df["pm_laxness"], df["pm_overactivity"]
    df["PrVrb"], df["PrSty"] = df["pm_verbosity"], df["pm_parent_style"]
    return df


MIMIC_COVARIATES = ["GA", "BirthWt", "Sex", "HdCirc", "IUGR", "GDiab", "PrEcl", "AntCor_P",
                    "AntCor_C", "MomAge", "MomEdu", "DadEdu", "MomEng", "EPDSTot", "StimEnv",
                    "PrLax", "PrOvr", "PrVrb", "PrSty"]


def compare_with_combined_dat(df: pd.DataFrame) -> pd.DataFrame:
    """The re-derived MIMIC covariates must equal the columns of combined.dat."""
    names = [l.strip() for l in (PSY / "varnames.txt").read_text().splitlines() if l.strip()]
    dat = np.loadtxt(PSY / "combined.dat")
    if dat.shape[0] != len(df):
        raise SystemExit(f"combined.dat has {dat.shape[0]} rows, the sample {len(df)}")
    rows = []
    for v in MIMIC_COVARIATES:
        col = dat[:, names.index(v)]
        col = np.where(col == -999, np.nan, col)
        mine = df[v].to_numpy(dtype=float)
        same_missing = np.array_equal(np.isnan(col), np.isnan(mine))
        ok = np.isfinite(col) & np.isfinite(mine)
        max_abs = float(np.nanmax(np.abs(col[ok] - mine[ok]))) if ok.any() else 0.0
        rows.append({"covariate": v, "n_combined_dat": int(np.isfinite(col).sum()),
                     "n_table1": int(np.isfinite(mine).sum()),
                     "same_missing_pattern": same_missing, "max_abs_diff": max_abs})
    cmp = pd.DataFrame(rows)
    cmp["match"] = (cmp.n_combined_dat == cmp.n_table1) & cmp.same_missing_pattern & (cmp.max_abs_diff < 1e-6)
    return cmp


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def make_table(df: pd.DataFrame) -> Table:
    n = len(df)
    t = Table(n)
    mimic = "MIMIC covariate {} ({})".format

    # --- infant characteristics ----------------------------------------------
    B = "Infant characteristics"
    sex = df["Sex"].dropna()
    t.n_pct(B, "Sex", "female", int((sex == 0).sum()), int(sex.size),
            mimic("Sex", "binary, STDY; coded 1 = male, 0 = female, so the Figure 4A effect is male vs female"))
    t.mean_sd(B, "Gestational age at birth, weeks", df["GA"],
              mimic("GA", "continuous, STDYX") + "; lpb01 baby_gest_at_birth, scan-derived fallback where missing")
    ga = df["GA"].dropna()
    t.n_pct(B, "Preterm birth", "<37 weeks", int((ga < 37).sum()), int(ga.size),
            "not a MIMIC covariate (GA entered continuously)")
    t.n_pct(B, "Very preterm birth", "<32 weeks", int((ga < 32).sum()), int(ga.size),
            "not a MIMIC covariate (GA entered continuously)")
    t.mean_sd(B, "Birth weight, kg", df["BirthWt"],
              mimic("BirthWt", "continuous, STDYX") + "; entered in kg (Figure 4A label says g; STDYX is scale-free)")
    bw = df["BirthWt"].dropna()
    t.n_pct(B, "Birth weight <1500 g", "<1500 g", int((bw < 1.5).sum()), int(bw.size),
            "of infants with birth weight; not a MIMIC covariate")
    cent = df["bw_centile"].dropna()
    t.n_pct(B, "Growth restriction, birth weight <10th INTERGROWTH-21st centile", "<10th centile",
            int((cent < 10).sum()), int(cent.size),
            "sex-specific INTERGROWTH-21st centile for gestational age (analysis/growth_restriction.py); "
            "denominator = infants with birth weight and GA within 24+0 to 42+6 weeks; not a MIMIC covariate")
    hc_raw, hc = df["HdCirc_raw"], df["HdCirc"]
    clipped = hc_raw[hc_raw.notna() & hc.isna()]
    n_sentinel = int((clipped == -996).sum())
    n_measured_removed = int(clipped.size - n_sentinel)
    t.mean_sd(B, "Head circumference at birth, cm", hc,
              mimic("HdCirc", "continuous, STDYX") + f"; lpb01 baby_babyhc ({int(hc_raw.notna().sum())} records); "
              f"the +/-{HC_SD_CLIP} SD rule of prepare_data.R removed {clipped.size} record(s): {n_sentinel} coded -996 "
              f"(an NDA missing-value code) and {n_measured_removed} measured value(s)")
    size = df["pregnancy_size"].dropna()
    for cat, mask in (("singleton", size == 1), ("twin", size == 2), ("higher-order", size >= 3)):
        t.n_pct(B, "Plurality", cat, int(mask.sum()), int(size.size),
                "cpenr01 pregnancy_size (at conception); manuscript percentages use all 739 as denominator; "
                "not a MIMIC covariate")
    nicu = df["baby_admitted_to_nicu"].dropna()
    t.n_pct(B, "NICU admission", "yes", int((nicu == 1).sum()), int(nicu.size),
            "lpb01 baby_admitted_to_nicu; manuscript reports 166/739 = 22.5% (all infants); not a MIMIC covariate")

    # --- perinatal and obstetric -----------------------------------------------
    B = "Perinatal and obstetric"
    ac = df["study_corticosteroids"].dropna()
    for cat, code, note in (("none", 0, "reference category of the MIMIC dummies AntCor_P / AntCor_C"),
                            ("partial course", 1, mimic("AntCor_P", "binary, STDY; 1 = partial course, 0 = none or complete")),
                            ("complete course", 2, mimic("AntCor_C", "binary, STDY; 1 = complete course, 0 = none or partial"))):
        t.n_pct(B, "Antenatal corticosteroids", cat, int((ac == code).sum()), int(ac.size),
                "lpb01 study_corticosteroids; " + note)
    iugr = df["IUGR"].dropna()
    t.n_pct(B, "Clinically suspected intrauterine growth restriction", "yes", int((iugr == 1).sum()), int(iugr.size),
            mimic("IUGR", "binary, STDY") + "; lpb01 baby_suspected_iugr")
    gd = df["GDiab"].dropna()
    t.n_pct(B, "Gestational diabetes", "yes", int((gd == 1).sum()), int(gd.size),
            mimic("GDiab", "binary, STDY") + "; lpb01 study_gestational_diabetes")
    pe_raw = df["study_preeclampsia"].dropna()
    pe = df["PrEcl"].dropna()
    codes = {int(k): int(v) for k, v in pe_raw.value_counts().sort_index().items()}
    t.n_pct(B, "Pre-eclampsia", "yes", int((pe == 1).sum()), int(pe.size),
            mimic("PrEcl", "binary, STDY") + "; lpb01 study_preeclampsia ('pre-eclampsia, HELLP or pregnancy-induced "
            f"hypertension') dichotomised 0 = none vs 1 = any of codes 1-3, as in prepare_data.R; raw code counts {codes}")

    # --- assessment -----------------------------------------------------------
    B = "Assessment"
    t.mean_sd(B, "Corrected age at assessment, months", df["corr_age"],
              "bsid_iii01 interview_age minus (40 - GA)/4.3482 months for infants born before 37 weeks "
              "(analysis/age_at_assessment.py)")
    t.mean_sd(B, "Chronological age at assessment, months", df["chron_age"], "bsid_iii01 interview_age")
    for lab, col in (("cognitive", "bsid_cog_composite"), ("language", "bsid_lang_composite"),
                     ("motor", "bsid_mot_composite")):
        t.mean_sd(B, f"Bayley-III {lab} composite score", gr.to_num(df[col]), f"bsid_iii01 {col}")

    # --- maternal and family --------------------------------------------------
    B = "Maternal and family"
    t.mean_sd(B, "Maternal age at birth, years", df["MomAge"],
              mimic("MomAge", "continuous, STDYX") + "; cpenr01 mother_age1 (age at expected delivery)")
    t.mean_sd(B, "Maternal age at leaving formal education, years", df["MomEdu"],
              mimic("MomEdu", "continuous, STDYX") + "; cpenr01 mother_education1 (age last in continuous full-time education)")
    t.mean_sd(B, "Paternal age at leaving formal education, years", df["DadEdu"],
              mimic("DadEdu", "continuous, STDYX") + "; cpenr01 father_education1 (age last in continuous full-time education)")
    eng = df["MomEng"].dropna()
    t.n_pct(B, "Maternal English as first language", "yes", int((eng == 1).sum()), int(eng.size),
            mimic("MomEng", "binary, STDY") + "; cpenr01 mother_eng_first_lan")
    t.mean_sd(B, "Edinburgh Postnatal Depression Scale total", df["EPDSTot"],
              mimic("EPDSTot", "continuous, STDYX") + "; epds01 epds_tot, earliest record per mother (prepare_data.R)")
    for lab, col in (("total", "PrSty"), ("laxness", "PrLax"), ("overreactivity", "PrOvr"), ("verbosity", "PrVrb")):
        src = {"PrSty": "pm_parent_style", "PrLax": "pm_laxness", "PrOvr": "pm_overactivity", "PrVrb": "pm_verbosity"}[col]
        t.mean_sd(B, f"Parenting Scale, {lab}", df[col],
                  mimic(col, "continuous, STDYX") + f"; pqmf01 {src}, earliest visit (prepare_data.R)")
    t.mean_sd(B, "Cognitively Stimulating Parenting Scale total", df["StimEnv"],
              mimic("StimEnv", "continuous, STDYX") + "; stps01 total_sps_home_environment")
    return t


# ---------------------------------------------------------------------------
# checks against the Participants paragraph
# ---------------------------------------------------------------------------
def participants_paragraph_checks(tab: pd.DataFrame) -> pd.DataFrame:
    """Expected values transcribed from the Participants paragraph, with the
    rounding the manuscript uses (1 decimal; birth weight 2; ranges as integers
    where the text gives integers)."""
    ix = tab.set_index(["variable", "category"])

    def row(var, cat=""):
        return ix.loc[(var, cat)]

    def pct_of(var, cat, denom):
        return 100 * row(var, cat)["value"] / denom

    n_total = int(row("Sex", "female")["n_with_data"])
    checks = [
        ("n", 739, len(tab) and n_total),
        ("female %", 46.8, row("Sex", "female")["sd_or_pct"]),
        ("GA mean", 38.3, row("Gestational age at birth, weeks")["value"]),
        ("GA SD", 3.8, row("Gestational age at birth, weeks")["sd_or_pct"]),
        ("GA min", 23, row("Gestational age at birth, weeks")["low"]),
        ("GA max", 43, row("Gestational age at birth, weeks")["high"]),
        ("preterm n", 158, row("Preterm birth", "<37 weeks")["value"]),
        ("preterm %", 21.4, row("Preterm birth", "<37 weeks")["sd_or_pct"]),
        ("<32 wk n", 69, row("Very preterm birth", "<32 weeks")["value"]),
        ("<32 wk %", 9.3, row("Very preterm birth", "<32 weeks")["sd_or_pct"]),
        ("<1500 g n", 60, row("Birth weight <1500 g", "<1500 g")["value"]),
        ("<1500 g denominator", 694, row("Birth weight <1500 g", "<1500 g")["n_with_data"]),
        ("<1500 g %", 8.6, row("Birth weight <1500 g", "<1500 g")["sd_or_pct"]),
        ("growth restriction n", 81, row("Growth restriction, birth weight <10th INTERGROWTH-21st centile", "<10th centile")["value"]),
        ("growth restriction denominator", 691, row("Growth restriction, birth weight <10th INTERGROWTH-21st centile", "<10th centile")["n_with_data"]),
        ("growth restriction %", 11.7, row("Growth restriction, birth weight <10th INTERGROWTH-21st centile", "<10th centile")["sd_or_pct"]),
        ("birth weight mean", 3.01, row("Birth weight, kg")["value"]),
        ("birth weight SD", 0.87, row("Birth weight, kg")["sd_or_pct"]),
        ("birth weight min", 0.45, row("Birth weight, kg")["low"]),
        ("birth weight max", 4.75, row("Birth weight, kg")["high"]),
        ("NICU n", 166, row("NICU admission", "yes")["value"]),
        ("NICU % (manuscript: of all 739)", 22.5, pct_of("NICU admission", "yes", 739)),
        ("NICU % (CSV: of the 694 with data)", 22.5, row("NICU admission", "yes")["sd_or_pct"]),
        ("singletons % (manuscript: of all 739)", 88.9, pct_of("Plurality", "singleton", 739)),
        ("singletons % (CSV: of the 737 with data)", 88.9, row("Plurality", "singleton")["sd_or_pct"]),
        ("twins %", 10.4, row("Plurality", "twin")["sd_or_pct"]),
        ("higher-order %", 0.4, row("Plurality", "higher-order")["sd_or_pct"]),
        ("corrected age mean", 19.4, row("Corrected age at assessment, months")["value"]),
        ("corrected age SD", 2.4, row("Corrected age at assessment, months")["sd_or_pct"]),
        ("corrected age min", 17.0, row("Corrected age at assessment, months")["low"]),
        ("corrected age max", 34.2, row("Corrected age at assessment, months")["high"]),
        ("chronological age mean", 19.8, row("Chronological age at assessment, months")["value"]),
        ("chronological age SD", 2.6, row("Chronological age at assessment, months")["sd_or_pct"]),
        ("chronological age min", 17, row("Chronological age at assessment, months")["low"]),
        ("chronological age max", 37, row("Chronological age at assessment, months")["high"]),
        ("cognitive composite mean", 100.8, row("Bayley-III cognitive composite score")["value"]),
        ("cognitive composite SD", 11.7, row("Bayley-III cognitive composite score")["sd_or_pct"]),
        ("language composite mean", 98.7, row("Bayley-III language composite score")["value"]),
        ("language composite SD", 16.2, row("Bayley-III language composite score")["sd_or_pct"]),
        ("motor composite mean", 101.6, row("Bayley-III motor composite score")["value"]),
        ("motor composite SD", 10.1, row("Bayley-III motor composite score")["sd_or_pct"]),
        ("maternal age mean", 34.2, row("Maternal age at birth, years")["value"]),
        ("maternal age SD", 4.6, row("Maternal age at birth, years")["sd_or_pct"]),
        ("maternal age min", 17, row("Maternal age at birth, years")["low"]),
        ("maternal age max", 52, row("Maternal age at birth, years")["high"]),
        ("left education mean", 23.3, row("Maternal age at leaving formal education, years")["value"]),
        ("left education SD", 4.4, row("Maternal age at leaving formal education, years")["sd_or_pct"]),
        ("left education min", 12, row("Maternal age at leaving formal education, years")["low"]),
        ("left education max", 41, row("Maternal age at leaving formal education, years")["high"]),
    ]
    out = []
    for label, exp, obs in checks:
        nd = 2 if isinstance(exp, float) and label.startswith("birth weight") else (1 if isinstance(exp, float) else 0)
        obs_r = round(float(obs), nd)                 # manuscript rounding of the unrounded value
        stored = ND + 1 if label.startswith("birth weight") else ND
        csv_r = round(round(float(obs), stored), nd)  # the same rounding applied to the stored CSV value
        out.append({"check": label, "manuscript": exp, "table1": obs_r, "ok": abs(obs_r - exp) < 1e-9,
                    "csv_double_rounding": csv_r != obs_r, "unrounded": float(obs)})
    return pd.DataFrame(out)


def main() -> int:
    df = build()
    table = make_table(df)
    tab = table.frame(rounded=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT, index=False)
    print(f"Table 1: {len(tab)} rows written to {OUT.relative_to(ROOT)}\n")

    print("Check 1 -- Participants paragraph vs Table 1 (unrounded values, manuscript rounding):")
    c1 = participants_paragraph_checks(table.frame())
    for r_ in c1.itertuples():
        nd = 2 if r_.check.startswith("birth weight") else 1
        warn = (f"   ! the stored CSV value rounds to "
                f"{round(round(r_.unrounded, ND), nd)}; format from the unrounded {r_.unrounded:.4f}"
                if r_.csv_double_rounding else "")
        print(f"  [{'ok  ' if r_.ok else 'DIFF'}] {r_.check:<42s} manuscript {r_.manuscript!s:>6}   table1 {r_.table1}{warn}")
    print(f"  {int(c1.ok.sum())}/{len(c1)} reproduced\n")

    print("Check 2 -- MIMIC covariates: Table 1 n_with_data vs non-missing cases in combined.dat:")
    c2 = compare_with_combined_dat(df)
    print(c2.to_string(index=False))
    if not c2.match.all():
        raise SystemExit("Table 1 covariates differ from combined.dat: " + ", ".join(c2.covariate[~c2.match]))
    print("  all covariates identical to combined.dat (n, missing pattern and values)\n")

    nicu = tab[tab.variable == "NICU admission"].iloc[0]
    print(f"Check 3 -- NICU: n = {nicu.value}, n_with_data = {nicu.n_with_data} "
          f"({nicu.sd_or_pct}% of those with data; manuscript 22.5% = {nicu.value}/739)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
