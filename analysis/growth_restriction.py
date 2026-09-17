"""
Birth-weight centiles and growth restriction (SGA) in the dHCP psychometric sample.

Reports the incidence of growth restriction (birth weight < 10th centile for
gestational age and sex) alongside the regulatory / descriptive birth-weight
cut-off (< 1500 g), so the association with gestational age can be contrasted
with a weight-based classification.

Standard
--------
INTERGROWTH-21st Newborn Size standards for weight-for-gestational-age, applied
sex-specifically:

  * 24+0 to 32+6 weeks (168-230 days): INTERGROWTH-21st Very Preterm Newborn
    Size reference (Villar et al., Lancet 2016;387:844-45).  log(birth weight in
    kg) ~ Normal(mu, sigma) with
        mu    = -7.00303 + 1.325911 * sqrt(GA_weeks) + 0.0571937 * male
        sigma = sqrt(0.0373218)

  * 33+0 to 42+6 weeks (231-300 days): INTERGROWTH-21st Newborn Size standard
    (Villar et al., Lancet 2014;384:857-68).  Skew-t type 3 (GAMLSS ST3)
    distribution with published mu/sigma/nu/tau coefficients per day of
    gestation and sex (ig_nbs_wfga_coeffs.csv).

The coefficient table and the very-preterm regression equations are those
distributed with the `gigs` R package (lshtm-gigs/gigs), which reproduces the
published INTERGROWTH-21st tables exactly.  Run with --validate DIR to re-derive
the published P03/P05/P10/P50/P90/P95/P97 tables from this code as a check.

Infants outside 24+0 to 42+6 weeks fall outside the standard and are reported
separately rather than extrapolated.

Usage
-----
    python analysis/growth_restriction.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
COGN_FILE = BASE / "data/processed/cogn_id_GA.xlsx"
LPB_FILE = BASE / "data/raw/dhcp_txt/lpb01.txt"
COEFF_FILE = Path(__file__).resolve().parent / "ig_nbs_wfga_coeffs.csv"

VPNS_MIN_DAYS = 168  # 24+0
NBS_MIN_DAYS = 231   # 33+0
NBS_MAX_DAYS = 300   # 42+6


# ---------------------------------------------------------------------------
# INTERGROWTH-21st weight-for-GA
# ---------------------------------------------------------------------------
def p_st3(q, mu, sigma, nu, tau):
    """CDF of the GAMLSS skew-t type 3 distribution (gamlss.dist::pST3)."""
    q, mu, sigma, nu, tau = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (q, mu, sigma, nu, tau)]
    )
    cdf1 = 2.0 * stats.t.cdf(nu * (q - mu) / sigma, df=tau)
    cdf2 = 1.0 + 2.0 * nu * nu * (stats.t.cdf((q - mu) / (sigma * nu), df=tau) - 0.5)
    return np.where(q < mu, cdf1, cdf2) / (1.0 + nu ** 2)


def q_st3(p, mu, sigma, nu, tau):
    """Quantile function of the GAMLSS skew-t type 3 (gamlss.dist::qST3)."""
    p, mu, sigma, nu, tau = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (p, mu, sigma, nu, tau)]
    )
    q1 = mu + (sigma / nu) * stats.t.ppf(p * (1 + nu ** 2) / 2.0, df=tau)
    q2 = mu + (sigma * nu) * stats.t.ppf(
        (p * (1 + nu ** 2) - 1.0) / (2.0 * nu ** 2) + 0.5, df=tau
    )
    return np.where(p < 1.0 / (1.0 + nu ** 2), q1, q2)


def vpns_mu_sigma(gest_days, male):
    """Very Preterm Newborn Size: mean/SD of log(birth weight in kg)."""
    ga_weeks = np.asarray(gest_days, dtype=float) / 7.0
    mu = -7.00303 + 1.325911 * np.sqrt(ga_weeks) + 0.0571937 * np.asarray(male, float)
    sigma = np.full_like(mu, np.sqrt(0.0373218))
    return mu, sigma


def load_coeffs() -> pd.DataFrame:
    return pd.read_csv(COEFF_FILE)


def bw_centile(weight_kg, gest_days, male, coeffs=None):
    """
    INTERGROWTH-21st birth-weight centile (0-1) and z-score.

    Returns (centile, zscore); both NaN outside 24+0 to 42+6 weeks or where an
    input is missing.
    """
    coeffs = load_coeffs() if coeffs is None else coeffs
    w = np.asarray(weight_kg, dtype=float)
    gd = np.asarray(gest_days, dtype=float)
    male = np.asarray(male, dtype=float)

    cent = np.full(w.shape, np.nan)
    ok = np.isfinite(w) & np.isfinite(gd) & np.isfinite(male) & (w > 0)

    # very preterm branch (24+0 to 32+6)
    vp = ok & (gd >= VPNS_MIN_DAYS) & (gd < NBS_MIN_DAYS)
    if vp.any():
        mu, sigma = vpns_mu_sigma(gd[vp], male[vp])
        cent[vp] = stats.norm.cdf((np.log(w[vp]) - mu) / sigma)

    # term / late-preterm branch (33+0 to 42+6)
    nb = ok & (gd >= NBS_MIN_DAYS) & (gd <= NBS_MAX_DAYS)
    if nb.any():
        tab = coeffs.set_index(["sex", "gest_days"])
        idx = pd.MultiIndex.from_arrays(
            [np.where(male[nb] == 1, "male", "female"), gd[nb].astype(int)]
        )
        sel = tab.loc[idx]
        cent[nb] = p_st3(
            w[nb], sel["mu"].to_numpy(), sel["sigma"].to_numpy(),
            sel["nu"].to_numpy(), sel["tau"].to_numpy(),
        )

    with np.errstate(invalid="ignore"):
        z = stats.norm.ppf(np.clip(cent, 1e-12, 1 - 1e-12))
    z[~np.isfinite(cent)] = np.nan
    return cent, z


def validate(coeffs: pd.DataFrame, tables_dir: Path) -> None:
    """Re-derive the published INTERGROWTH-21st centile tables from this code."""
    targets = {"P03": 0.03, "P05": 0.05, "P10": 0.10, "P50": 0.50,
               "P90": 0.90, "P95": 0.95, "P97": 0.97}
    for sex, male in (("male", 1.0), ("female", 0.0)):
        pub = pd.read_csv(tables_dir / f"w_{sex}_pct.csv", header=None,
                          sep=r"\s+")
        pub.columns = ["ga"] + list(targets)
        parts = pub["ga"].str.split("+", expand=True).astype(int)
        gd = (parts[0] * 7 + parts[1]).to_numpy()
        for name, p in targets.items():
            pred = np.full(gd.shape, np.nan, dtype=float)
            vp = gd < NBS_MIN_DAYS
            mu, sigma = vpns_mu_sigma(gd[vp], male)
            pred[vp] = np.exp(mu + sigma * stats.norm.ppf(p))
            sel = coeffs[coeffs["sex"] == sex].set_index("gest_days").loc[gd[~vp]]
            pred[~vp] = q_st3(p, sel["mu"].to_numpy(), sel["sigma"].to_numpy(),
                              sel["nu"].to_numpy(), sel["tau"].to_numpy())
            diff = np.abs(np.round(pred, 2) - pub[name].to_numpy())
            print(f"  {sex:6s} {name}: max |published - reproduced| = "
                  f"{np.nanmax(diff):.3f} kg over {len(gd)} GA days")


# ---------------------------------------------------------------------------
# dHCP sample
# ---------------------------------------------------------------------------
def read_ndar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    return df.iloc[1:]  # drop the NDAR value-range row


def to_num(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    return n.mask(n.isin([-999, -998, 999]))


def build_sample() -> pd.DataFrame:
    """The 739-infant psychometric sample with perinatal anthropometry."""
    cogn = pd.read_excel(COGN_FILE)[
        ["src_subject_id", "sex", "lpb_ga_at_birth_weeks", "nscan_ga_at_birth_weeks"]
    ]
    lpb = read_ndar(LPB_FILE)[
        ["src_subject_id", "baby_gest_at_birth", "baby_birth_weight",
         "baby_gender", "baby_suspected_iugr", "baby_admitted_to_nicu"]
    ].copy()
    for c in lpb.columns[1:]:
        lpb[c] = to_num(lpb[c])
    lpb = lpb.drop_duplicates("src_subject_id")

    df = cogn.merge(lpb, on="src_subject_id", how="left")

    # GA: lpb01 primary, scan-derived fallback (matches prepare_data.R)
    df["GA"] = df["baby_gest_at_birth"].fillna(to_num(df["nscan_ga_at_birth_weeks"]))
    df["gest_days"] = np.round(df["GA"] * 7)
    df["BirthWt"] = df["baby_birth_weight"]
    # Sex: prefer lpb01 baby_gender (1 = M, 2 = F), fall back to the cogn file
    male = np.where(df["baby_gender"] == 1, 1.0,
                    np.where(df["baby_gender"] == 2, 0.0, np.nan))
    df["male"] = np.where(np.isfinite(male), male, (df["sex"] == "M").astype(float))
    df["IUGR"] = df["baby_suspected_iugr"]
    return df


def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)" if d else f"{n} (-)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", metavar="DIR", type=Path, default=None,
                    help="directory holding w_male_pct.csv / w_female_pct.csv")
    ap.add_argument("--out", type=Path,
                    default=BASE / "data/outputs/growth_restriction_per_infant.csv")
    args = ap.parse_args()

    coeffs = load_coeffs()
    if args.validate:
        print("Validation against the published INTERGROWTH-21st tables:")
        validate(coeffs, args.validate)
        print()

    df = build_sample()
    cent, z = bw_centile(df["BirthWt"], df["gest_days"], df["male"], coeffs)
    df["bw_centile"] = cent * 100
    df["bw_z"] = z
    df["SGA10"] = df["bw_centile"] < 10
    df["SGA3"] = df["bw_centile"] < 3
    df["preterm"] = df["GA"] < 37
    df["vlbw"] = df["BirthWt"] < 1.5

    n = len(df)
    have_bw = df["BirthWt"].notna()
    scored = df["bw_centile"].notna()
    oor = have_bw & ~scored

    print(f"Sample: N = {n}")
    print(f"  Birth weight recorded : {pct(int(have_bw.sum()), n)}")
    print(f"  Centile assignable    : {pct(int(scored.sum()), n)}"
          f"  [of those with birth weight: {pct(int(scored.sum()), int(have_bw.sum()))}]")
    if oor.any():
        print(f"  Outside 24+0-42+6 wk  : {int(oor.sum())} "
              f"(GA = {sorted(df.loc[oor, 'GA'].round(2).tolist())})")

    d = int(scored.sum())
    print("\nGrowth restriction (INTERGROWTH-21st weight-for-GA, sex-specific):")
    print(f"  Birth weight < 10th centile (SGA): "
          f"{pct(int(df.loc[scored, 'SGA10'].sum()), d)}")
    print(f"  Birth weight <  3rd centile      : "
          f"{pct(int(df.loc[scored, 'SGA3'].sum()), d)}")
    print(f"  Birth-weight z-score: M = {df['bw_z'].mean():.2f}, "
          f"SD = {df['bw_z'].std():.2f}, "
          f"range = {df['bw_z'].min():.2f} to {df['bw_z'].max():.2f}")

    print("\nWeight-based cut-off:")
    print(f"  Birth weight < 1500 g (VLBW)     : "
          f"{pct(int(df.loc[have_bw, 'vlbw'].sum()), int(have_bw.sum()))}")

    print("\nSGA by gestational age group (infants with an assignable centile):")
    for label, mask in (("Preterm (<37 wk)", df["preterm"] == True),
                        ("Term (>=37 wk)", df["preterm"] == False)):
        m = scored & mask
        print(f"  {label:18s} n = {int(m.sum()):3d}, "
              f"SGA = {pct(int(df.loc[m, 'SGA10'].sum()), int(m.sum()))}")

    print("\nOverlap of the two classifications:")
    both = scored & have_bw
    print(pd.crosstab(df.loc[both, "SGA10"], df.loc[both, "vlbw"],
                      rownames=["SGA <10th"], colnames=["BW <1500 g"]).to_string())

    print("\nClinically suspected IUGR (lpb01 baby_suspected_iugr):")
    iu = df["IUGR"].notna()
    print(f"  Recorded : {pct(int(iu.sum()), n)}")
    print(f"  Suspected: {pct(int(df.loc[iu, 'IUGR'].sum()), int(iu.sum()))}")
    agree = iu & scored
    print("  Cross-tabulation with SGA:")
    print(pd.crosstab(df.loc[agree, "IUGR"].astype(int), df.loc[agree, "SGA10"],
                      rownames=["Suspected IUGR"], colnames=["SGA <10th"]).to_string())

    cols = ["src_subject_id", "GA", "gest_days", "male", "BirthWt",
            "bw_centile", "bw_z", "SGA10", "SGA3", "vlbw", "preterm", "IUGR"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.out, index=False)
    print(f"\nPer-infant values written to {args.out.relative_to(BASE)}")


if __name__ == "__main__":
    main()
