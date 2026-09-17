"""
check_columns.py — Verify combined.dat's column count matches varnames.txt,
and sanity-check a handful of sentinel columns against their expected
variable name and value range.

This is the structural counterpart to check_integrity.py: it catches a
"NAMES ARE list out of sync with combined.dat width" error, which would
silently misalign the columns in every .inp file reading combined.dat.

Run from repo root:
  python psychometrics/checks/check_columns.py
"""

from pathlib import Path

import numpy as np

MISSING = -999
PSYCHOMETRICS_DIR = Path(__file__).resolve().parents[1]

with open(PSYCHOMETRICS_DIR / "varnames.txt") as f:
    varnames = [l.strip() for l in f if l.strip()]

dat = np.loadtxt(PSYCHOMETRICS_DIR / "combined.dat", dtype=float)
n_rows, n_cols = dat.shape
print(f"combined.dat: {n_rows} rows x {n_cols} cols")
print(f"varnames.txt: {len(varnames)} names\n")

if n_cols != len(varnames):
    print(f"!!! MISMATCH: {n_cols} cols vs {len(varnames)} names !!!")
else:
    print("Column count matches varnames.txt\n")

sentinels = [
    (35, "GA",       "gestational age ~24-42 wks"),
    (37, "Sex",      "binary 0/1"),
    (44, "MomAge",   "maternal age ~20-45 yrs"),
    (51, "MomEng",   "binary 0/1"),
    (52, "EPDSTot",  "EPDS score 0-30"),
    (53, "StimEnv",  "home environment, continuous"),
    (54, "PrLax",    "parenting laxness, continuous"),
    (57, "PrSty",    "parenting style, continuous"),
    (58, "CBInt",    "CBCL internalising, continuous"),
    (66, "CBAgg",    "CBCL aggressive, continuous"),
    (70, "BayMot",   "Bayley motor composite"),
    (71, "BayLan",   "Bayley language composite"),
    (72, "QchatTot", "Q-CHAT total, continuous"),
]

hdr = f"{'Col':>3}  {'Expected':12}  {'In .dat':12}  {'n_valid':>7}  {'min':>7}  {'max':>7}  {'mean':>7}  Expected range"
print(hdr)
print("-" * len(hdr))
for col1, expected_name, note in sentinels:
    col0 = col1 - 1
    actual_name = varnames[col0]
    vals = dat[:, col0]
    valid = vals[vals != MISSING]
    mn = valid.min() if len(valid) else float('nan')
    mx = valid.max() if len(valid) else float('nan')
    mu = valid.mean() if len(valid) else float('nan')
    ok = "OK" if actual_name == expected_name else "!!! MISMATCH !!!"
    print(f"{col1:>3}  {expected_name:12}  {actual_name:12}  {len(valid):>7}  {mn:>7.2f}  {mx:>7.2f}  {mu:>7.2f}  {note}  {ok}")
