"""
Diagnose missing data in combined.dat (Mplus input file).
Missing sentinel = -999.

Run from repo root:
  python psychometrics/checks/check_missing_data.py
"""
from pathlib import Path

import numpy as np
from collections import Counter

PSYCHOMETRICS_DIR = Path(__file__).resolve().parents[1]

# ── load ──────────────────────────────────────────────────────────────────────
with open(PSYCHOMETRICS_DIR / "varnames.txt") as f:
    varnames = [l.strip() for l in f if l.strip()]

data = np.loadtxt(PSYCHOMETRICS_DIR / "combined.dat")
n, p = data.shape
print(f"Shape: {n} subjects × {p} variables")
print()

missing_mask = data == -999

# ── per-variable summary ──────────────────────────────────────────────────────
print("=== Missing count per variable ===")
demo_vars = ["GA","BirthWt","Sex","HdCirc","IUGR","GDiab","PrEcl","AntCor_P","AntCor_C",
             "MomAge","MomEdu","DadEdu","EthSAsian","EthBlack","EthMixed","EthOther","MomEng",
             "EPDSTot","StimEnv","PrLax","PrOvr","PrVrb","PrSty"]
demo_vars = [v for v in demo_vars if v in varnames]

# Show all non-item variables
item_prefix_idx = max(i for i,v in enumerate(varnames) if v.startswith("COG")) + 1
non_item_vars = varnames[item_prefix_idx:]

for v in non_item_vars:
    idx = varnames.index(v)
    n_miss = int(missing_mask[:, idx].sum())
    flag = " *** DEMO ***" if v in demo_vars else ""
    print(f"  {v:14s}  missing: {n_miss:4d} / {n}  ({100*n_miss/n:.1f}%){flag}")

print()
# ── subject-level demo completeness ──────────────────────────────────────────
demo_indices = [varnames.index(v) for v in demo_vars]
demo_miss = missing_mask[:, demo_indices]  # shape (n, 13)
any_miss = demo_miss.any(axis=1)
print(f"Subjects with ≥1 missing demographic : {any_miss.sum():4d} / {n}  ({100*any_miss.mean():.1f}%)")
print(f"Subjects complete on ALL 13 covariates: {(~any_miss).sum():4d} / {n}  ({100*(~any_miss).mean():.1f}%)")

print()
# ── missing patterns ─────────────────────────────────────────────────────────
print("Top missing-data patterns (which demo vars are absent together):")
patterns = []
for row in demo_miss[any_miss]:
    patterns.append(tuple(v for v, m in zip(demo_vars, row) if m))
cnt = Counter(patterns)
for pattern, count in cnt.most_common(15):
    print(f"  {count:4d}x  {list(pattern)}")

print()
# ── item missingness ─────────────────────────────────────────────────────────
item_vars = [v for v in varnames if v.startswith("COG")]
item_indices = [varnames.index(v) for v in item_vars]
item_miss = missing_mask[:, item_indices]
any_item_miss = item_miss.any(axis=1)
print(f"Subjects with ≥1 missing item      : {any_item_miss.sum():4d} / {n}")
print(f"Subjects with ALL items present     : {(~any_item_miss).sum():4d} / {n}")

print()
# ── cross-tab: demo-complete vs item-complete ─────────────────────────────────
both_complete = (~any_miss) & (~any_item_miss)
demo_only_miss = any_miss & (~any_item_miss)
print(f"Both items + demos complete : {both_complete.sum()}")
print(f"Items OK but demos missing  : {demo_only_miss.sum()}")
