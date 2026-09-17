#!/usr/bin/env python
"""
Cohort counts and study dates of the dHCP release used in the paper.

Reads the restricted NDA instruments (docs/DATA_ACCESS.md) and writes
results/metrics/cohort_counts.json with

  * the number of distinct subject IDs across all instruments in
    data/raw/dhcp_txt/ (the fourth-release cohort available to us), and the
    number of subjects per instrument;
  * the number of subjects with a bsid_iii01 record (the 18-month Bayley-III
    assessment);
  * the number of infants with item-level cognition responses in
    data/processed/cogn_id_GA.xlsx (the analysis sample, N = 739);
  * the recruitment period = range of `interview_date` on the enrolment
    record (cpenr01; identical to ndar_subject01), and the dates of the
    18-month assessments = range of `interview_date` in bsid_iii01, each for
    the whole cohort and for the 739 analysed infants.

Only aggregate counts and date ranges are written; no subject-level data.
The values are checked by analysis/verify_manuscript_numbers.py ("Cohort").

Usage (from the repository root):
    python analysis/cohort_counts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DHCP = ROOT / "data/raw/dhcp_txt"
COGN = ROOT / "data/processed/cogn_id_GA.xlsx"
OUT = ROOT / "results/metrics/cohort_counts.json"

ENROLMENT = "cpenr01"       # enrolment record: one row per subject, interview_age 0
BAYLEY = "bsid_iii01"       # 18-month Bayley-III assessment
DATE_FORMAT = "%m/%d/%Y"    # NDA interview_date


def read_ndar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    return df.iloc[1:]          # drop the NDA value-range row


def date_range(df: pd.DataFrame, mask=None) -> dict:
    dt = pd.to_datetime(df["interview_date"], errors="coerce", format=DATE_FORMAT)
    if dt.isna().any():
        raise SystemExit(f"{int(dt.isna().sum())} interview_date value(s) not in {DATE_FORMAT}")
    if mask is not None:
        dt = dt[mask]
    return {"first": str(dt.min().date()), "last": str(dt.max().date()), "n": int(dt.notna().sum())}


def main() -> int:
    frames = {p.stem: read_ndar(p) for p in sorted(DHCP.glob("*.txt"))}
    ids_per_instrument = {k: set(v["src_subject_id"].dropna()) for k, v in frames.items()}
    all_ids = set().union(*ids_per_instrument.values())
    analysed = set(pd.read_excel(COGN)["src_subject_id"])
    bayley_ids = ids_per_instrument[BAYLEY]

    enrol = frames[ENROLMENT]
    bayley = frames[BAYLEY]
    out = {
        "instruments": sorted(frames),
        "n_subjects_all_instruments": len(all_ids),
        "n_subjects_per_instrument": {k: len(v) for k, v in ids_per_instrument.items()},
        "n_rows_per_instrument": {k: int(len(v)) for k, v in frames.items()},
        "n_subjects_bsid_iii01": len(bayley_ids),
        "n_infants_item_level_cognition": len(analysed),
        "analysed_infants_with_bsid_record": len(analysed & bayley_ids),
        "analysed_infants_in_instruments": len(analysed & all_ids),
        "recruitment_period": {
            "source": f"{ENROLMENT}.interview_date (enrolment record, interview_age 0)",
            "all_subjects": date_range(enrol),
            "analysed_infants": date_range(enrol, enrol["src_subject_id"].isin(analysed)),
        },
        "bayley_assessment_dates": {
            "source": f"{BAYLEY}.interview_date (18-month Bayley-III visit)",
            "all_subjects": date_range(bayley),
            "analysed_infants": date_range(bayley, bayley["src_subject_id"].isin(analysed)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rp, bd = out["recruitment_period"], out["bayley_assessment_dates"]
    print(f"Distinct subjects across the {len(frames)} instruments: {out['n_subjects_all_instruments']}")
    print(f"Subjects with a bsid_iii01 record (18-month Bayley-III): {out['n_subjects_bsid_iii01']}")
    print(f"Infants with item-level cognition responses (cogn_id_GA.xlsx): {out['n_infants_item_level_cognition']} "
          f"(all with a bsid_iii01 record: {out['analysed_infants_with_bsid_record']})")
    print(f"Recruitment period ({rp['source']}): {rp['all_subjects']['first']} to {rp['all_subjects']['last']} "
          f"(all {rp['all_subjects']['n']}); analysed infants {rp['analysed_infants']['first']} to "
          f"{rp['analysed_infants']['last']}")
    print(f"18-month assessments ({bd['source']}): {bd['all_subjects']['first']} to {bd['all_subjects']['last']} "
          f"(n={bd['all_subjects']['n']})")
    print(f"\nWritten to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
