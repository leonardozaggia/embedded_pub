#!/usr/bin/env python
"""
Sample descriptives reported in the Methods ("Sample") of the manuscript.

Reproduces every number of that paragraph from the restricted dHCP / NDA
instruments (docs/DATA_ACCESS.md) and writes them to
results/metrics/sample_descriptives.json:

  * N, % female                              bsid_iii01, lpb01 (baby_gender)
  * gestational age (GA) M / SD / range,     lpb01 (baby_gest_at_birth, scan-derived
    preterm (<37 wk), very preterm (<32 wk)   fallback from data/processed/cogn_id_GA.xlsx)
  * birth weight M / SD / range, <1500 g     lpb01 (baby_birth_weight, kg)
  * growth restriction (birth weight <10th   analysis/growth_restriction.py
    INTERGROWTH-21st centile for GA and sex)  (INTERGROWTH-21st standards)
  * NICU admission                           lpb01 (baby_admitted_to_nicu)
  * singletons / twins / higher-order        cpenr01 (pregnancy_size)
  * chronological and corrected age at       analysis/age_at_assessment.py
    the Bayley-III assessment
  * Bayley-III cognitive, language and       bsid_iii01 (*_composite)
    motor composites
  * maternal age, age at leaving education   cpenr01 (mother_age1, mother_education1)

Usage (from the repository root):
    python analysis/sample_descriptives.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import growth_restriction as gr      # noqa: E402  (INTERGROWTH-21st centiles)
import age_at_assessment as aa       # noqa: E402  (corrected age)

DHCP = ROOT / "data/raw/dhcp_txt"
OUT = ROOT / "results/metrics/sample_descriptives.json"


def msd(x: pd.Series, nd: int = 1) -> dict:
    x = x.dropna()
    return {"n": int(x.size), "mean": round(float(x.mean()), nd), "sd": round(float(x.std()), nd),
            "min": round(float(x.min()), nd), "max": round(float(x.max()), nd)}


def frac(n: int, d: int) -> dict:
    return {"n": int(n), "of": int(d), "pct": round(100 * n / d, 1)}


def main() -> int:
    # --- perinatal sample (739 infants with Bayley-III item data) ------------
    df = gr.build_sample()
    n = len(df)
    coeffs = gr.load_coeffs()
    cent, z = gr.bw_centile(df["BirthWt"], df["gest_days"], df["male"], coeffs)
    df["bw_centile"] = cent * 100
    sga = df["bw_centile"] < 10
    scored = df["bw_centile"].notna()
    have_bw = df["BirthWt"].notna()

    out = {
        "n_infants": n,
        "female": frac(int((df["male"] == 0).sum()), n),
        "gestational_age_weeks": msd(df["GA"]),
        "preterm_lt37": frac(int((df["GA"] < 37).sum()), n),
        "very_preterm_lt32": frac(int((df["GA"] < 32).sum()), n),
        "birth_weight_kg": msd(df["BirthWt"], 2),
        "birth_weight_lt1500g": frac(int((df["BirthWt"] < 1.5).sum()), int(have_bw.sum())),
        "growth_restriction_sga10": {
            **frac(int(sga.sum()), int(scored.sum())),
            "of_whom_lt1500g": int((sga & (df["BirthWt"] < 1.5)).sum()),
            "definition": "birth weight < 10th sex-specific INTERGROWTH-21st centile for gestational age; "
                          "infants outside 24+0 to 42+6 weeks or without birth weight are not eligible",
        },
    }

    # --- NICU admission ------------------------------------------------------
    nicu = df["baby_admitted_to_nicu"]
    out["nicu_admission"] = {
        "n": int((nicu == 1).sum()), "n_with_data": int(nicu.notna().sum()),
        "pct_of_all": round(100 * (nicu == 1).sum() / n, 1),
        "pct_of_those_with_data": round(100 * (nicu == 1).sum() / nicu.notna().sum(), 1),
    }

    # --- plurality (pregnancy size at conception) -----------------------------
    cpen = gr.read_ndar(DHCP / "cpenr01.txt").drop_duplicates("src_subject_id")
    cpen = cpen[cpen["src_subject_id"].isin(df["src_subject_id"])]
    size = gr.to_num(cpen["pregnancy_size"])
    out["plurality"] = {
        "singleton": frac(int((size == 1).sum()), n),
        "twin": frac(int((size == 2).sum()), n),
        "higher_order": frac(int((size >= 3).sum()), n),
        "missing": int(size.isna().sum()),
        "source": "cpenr01.pregnancy_size (pregnancy size at conception)",
    }

    # --- maternal age / education -------------------------------------------
    out["maternal_age_years"] = msd(gr.to_num(cpen["mother_age1"]))
    out["maternal_age_left_education_years"] = msd(gr.to_num(cpen["mother_education1"]))

    # --- age at assessment and composites -------------------------------------
    ages = aa.build()
    out["age_at_assessment_months"] = {
        "corrected": msd(ages["corr_age"]),
        "chronological": msd(ages["chron_age"]),
        "correction": "(40 - GA)/4.3482 months subtracted for infants born before 37 weeks",
    }
    out["bayley_composites"] = {
        "cognitive": msd(aa.to_num(ages["bsid_cog_composite"])),
        "language": msd(aa.to_num(ages["bsid_lang_composite"])),
        "motor": msd(aa.to_num(ages["bsid_mot_composite"])),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # --- printout in the order of the manuscript paragraph -------------------
    ga, bw = out["gestational_age_weeks"], out["birth_weight_kg"]
    f = out["female"]; pt = out["preterm_lt37"]; vpt = out["very_preterm_lt32"]
    vl = out["birth_weight_lt1500g"]; s = out["growth_restriction_sga10"]; ni = out["nicu_admission"]
    pl = out["plurality"]; ca = out["age_at_assessment_months"]["corrected"]
    ch = out["age_at_assessment_months"]["chronological"]; bc = out["bayley_composites"]
    ma, me = out["maternal_age_years"], out["maternal_age_left_education_years"]
    print(f"N = {n} infants ({f['pct']}% female)")
    print(f"GA {ga['min']:.0f}-{ga['max']:.0f} weeks (M = {ga['mean']}, SD = {ga['sd']}); "
          f"{pt['n']} preterm ({pt['pct']}%), {vpt['n']} < 32 weeks ({vpt['pct']}%), "
          f"{vl['n']} of {vl['of']} with birth weight data ({vl['pct']}%) < 1500 g")
    print(f"Growth restriction (birth weight < 10th INTERGROWTH-21st centile): {s['n']} of {s['of']} "
          f"eligible ({s['pct']}%), of whom {s['of_whom_lt1500g']} < 1500 g")
    print(f"Birth weight M = {bw['mean']:.2f} kg (SD = {bw['sd']:.2f}, range {bw['min']:.2f}-{bw['max']:.2f})")
    print(f"NICU admission: {ni['n']} infants ({ni['pct_of_all']}% of all {n}; "
          f"{ni['pct_of_those_with_data']}% of the {ni['n_with_data']} with data)")
    print(f"Singletons {pl['singleton']['pct']}% (twins {pl['twin']['pct']}%, higher-order {pl['higher_order']['pct']}%)")
    print(f"Corrected age at assessment M = {ca['mean']} months (SD = {ca['sd']}, range {ca['min']}-{ca['max']}); "
          f"chronological M = {ch['mean']} (SD = {ch['sd']}, range {ch['min']:.0f}-{ch['max']:.0f})")
    print(f"Composites: cognitive M = {bc['cognitive']['mean']} (SD = {bc['cognitive']['sd']}); "
          f"language M = {bc['language']['mean']} (SD = {bc['language']['sd']}); "
          f"motor M = {bc['motor']['mean']} (SD = {bc['motor']['sd']})")
    print(f"Mothers: age M = {ma['mean']} years (SD = {ma['sd']}, range {ma['min']:.0f}-{ma['max']:.0f}); "
          f"left education at M = {me['mean']} years (SD = {me['sd']}, range {me['min']:.0f}-{me['max']:.0f})")
    print(f"\nWritten to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
