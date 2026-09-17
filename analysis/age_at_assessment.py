"""
Chronological vs corrected age at Bayley-III assessment in the dHCP sample.

In a cohort enriched for preterm birth, age at assessment is reported
corrected for prematurity as well as chronologically.

What the NDAR release contains
------------------------------
`bsid_iii01.txt` has `interview_age` ("age in months at the time of the
interview") and `age_adj_premature` ("age adjusted for prematurity"), the latter
set to 1 for every infant.  `interview_age` is CHRONOLOGICAL: regressed on
gestational age it has a slope of about -0.23 months per week of GA, which is
what a fixed corrected-age assessment window implies ((40 - GA) / 4.35).  After
correction the slope collapses to ~0 (see `report()` below), confirming that
assessments were scheduled at a fixed corrected age.

Correction
----------
Bayley-III / AAP convention: subtract (40 - GA_weeks) / 4.3482 months, applied
only to infants born before 37 weeks.  A continuous variant that corrects every
infant to 40 weeks is also reported for comparison.

Usage
-----
    python psychometrics/checks/age_at_assessment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
BSID_FILE = BASE / "data/raw/dhcp_txt/bsid_iii01.txt"
LPB_FILE = BASE / "data/raw/dhcp_txt/lpb01.txt"
COGN_FILE = BASE / "data/processed/cogn_id_GA.xlsx"

WEEKS_PER_MONTH = 365.25 / 12 / 7  # 4.3482
PRETERM_WEEKS = 37


def read_ndar(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", skiprows=[1], low_memory=False)


def to_num(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    return n.mask(n.isin([-999, -998, 999]))


def build() -> pd.DataFrame:
    bsid = read_ndar(BSID_FILE)[
        ["src_subject_id", "interview_age", "age_adj_premature",
         "bsid_cog_composite", "bsid_lang_composite", "bsid_mot_composite"]
    ]
    lpb = read_ndar(LPB_FILE)[["src_subject_id", "baby_gest_at_birth"]]
    lpb = lpb.drop_duplicates("src_subject_id")
    cogn = pd.read_excel(COGN_FILE)[["src_subject_id", "lpb_ga_at_birth_weeks"]]

    df = bsid.merge(lpb, on="src_subject_id", how="left").merge(
        cogn, on="src_subject_id", how="left")
    df["GA"] = to_num(df["baby_gest_at_birth"]).fillna(
        to_num(df["lpb_ga_at_birth_weeks"]))
    df["chron_age"] = to_num(df["interview_age"])

    # Bayley/AAP convention: correct only infants born before 37 weeks
    adj = np.where(df["GA"] < PRETERM_WEEKS,
                   (40 - df["GA"]) / WEEKS_PER_MONTH, 0.0)
    df["corr_age"] = df["chron_age"] - adj
    df["adjustment"] = adj
    # continuous variant: correct every infant to 40 weeks
    df["corr_age_all"] = df["chron_age"] - np.maximum(0, 40 - df["GA"]) / WEEKS_PER_MONTH
    df["preterm"] = df["GA"] < PRETERM_WEEKS
    return df


def describe(x: pd.Series, label: str) -> None:
    print(f"  {label:38s} M={x.mean():.1f}  SD={x.std():.1f}  "
          f"median={x.median():.1f}  range={x.min():.1f}-{x.max():.1f}")


def report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"N = {n}")
    flag = df["age_adj_premature"].dropna().unique()
    print(f"  age_adj_premature values in the release: {sorted(flag)}\n")

    print("Age at Bayley-III assessment (months):")
    describe(df["chron_age"], "Chronological")
    describe(df["corr_age"], "Corrected (Bayley: <37 wk only)")
    describe(df["corr_age_all"], "Corrected (all infants, to 40 wk)")

    print("\nIs interview_age chronological or already corrected?")
    ok = df["GA"].notna() & df["chron_age"].notna()
    for col, label in (("chron_age", "chronological"), ("corr_age", "corrected")):
        fit = stats.linregress(df.loc[ok, "GA"], df.loc[ok, col])
        print(f"  {label:14s} ~ GA: slope = {fit.slope:+.3f} months/week "
              f"(r = {fit.rvalue:+.3f}, p = {fit.pvalue:.2g})")
    print(f"  expected slope if chronological and testing is at a fixed "
          f"corrected age: {-1 / WEEKS_PER_MONTH:+.3f}")

    print("\nBy gestational age group:")
    for label, mask in (("Term (>=37 wk)", df["GA"] >= PRETERM_WEEKS),
                        ("Preterm (<37 wk)", df["GA"] < PRETERM_WEEKS),
                        ("Very preterm (<32 wk)", df["GA"] < 32)):
        print(f"  {label:22s} n = {int(mask.sum()):3d}   "
              f"chronological M = {df.loc[mask, 'chron_age'].mean():.1f} "
              f"(SD {df.loc[mask, 'chron_age'].std():.1f})   "
              f"corrected M = {df.loc[mask, 'corr_age'].mean():.1f} "
              f"(SD {df.loc[mask, 'corr_age'].std():.1f})")

    pm = df["preterm"] == True
    print(f"\nCorrection applied to the {int(pm.sum())} preterm infants: "
          f"M = {df.loc[pm, 'adjustment'].mean():.1f} months "
          f"(range {df.loc[pm, 'adjustment'].min():.1f}-"
          f"{df.loc[pm, 'adjustment'].max():.1f})")
    print(f"  preterm infants assessed after 24 months chronological "
          f"(beyond the age at which Bayley-III correction is conventionally "
          f"applied): {int((pm & (df['chron_age'] > 24)).sum())}")

    print("\nComposite scores vs age at assessment:")
    for col, label in (("bsid_cog_composite", "Cognitive"),
                       ("bsid_lang_composite", "Language"),
                       ("bsid_mot_composite", "Motor")):
        v = to_num(df[col])
        m = v.notna() & df["corr_age"].notna()
        rc = stats.pearsonr(df.loc[m, "chron_age"], v[m])
        rk = stats.pearsonr(df.loc[m, "corr_age"], v[m])
        print(f"  {label:10s} chronological r = {rc[0]:+.3f} (p = {rc[1]:.2g})   "
              f"corrected r = {rk[0]:+.3f} (p = {rk[1]:.2g})")


if __name__ == "__main__":
    report(build())
