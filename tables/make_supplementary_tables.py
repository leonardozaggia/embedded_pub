#!/usr/bin/env python
"""
Build Supplementary Tables S1-S8 from the committed pipeline and Mplus outputs.

  Table S1  cascade Bayley-4 -> Bayley-III item pairs (39 rows), with the
            Aylward domain(s) of each Bayley-4 item and the first-listed domain
            under which it was transferred
            <- data/outputs/step1_translation/b4_b3_cascade_pairs.csv,
               config/reference_models_b4.yaml
  Table S2  every Bayley-III item administered in the dHCP window (items 34-68)
            with response frequencies and floor/ceiling status (35 rows)
            <- psychometrics/combined.dat, varnames.txt, factor_item_map.csv,
               measurement/model_cfa_3factor.inp
  Table S3  the three-factor CFA vs the unidimensional model
            a) standardised (STDYX) loadings of the 23 items, three-factor model
            b) inter-factor correlations
            c) global fit of both models (chi-square, df, RMSEA + 90% CI, CFI, TLI, SRMR)
               and the WLSMV chi-square difference test of the nested unidimensional
               model against the three-factor model (Mplus DIFFTEST)
            d) standardised loadings, unidimensional model
            <- psychometrics/measurement/model_cfa_3factor.out, model_unidimensional.out,
               model_unidimensional_difftest.out
  Table S4  MIMIC covariate associations (three factors + unidimensional composite)
            a) standardised regression coefficients -- continuous covariates
               fully standardised (STDYX), binary covariates semi-standardised
               (STDY), i.e. the convention used in the Results and Figure 4A
            b) WLSMV chi-square difference tests (DIFFTEST) for the factor contrasts
            <- psychometrics/mimic/mimic_std_effects.csv, difftest_results.csv
  Table S5  latent correlations between the factors and the concurrent
            developmental / behavioural outcomes (STDY)
            <- psychometrics/outcomes/outcomes_associations.csv
  Table S6  per-domain recovery of the Aylward domains by the Bayley-4 clusters,
            any-factor and single-domain rules (Results: 24/39, ATT 9/9, FS 9/10)
            <- config/reference_models_b4.yaml, step-2 item_comparison_theoretical.csv
  Table S7  pair-level position of the Bayley-4 clusters between the two
            reference models (consensus-apart / -together / disputed pairs) for
            three treatments of the multi-factor items -- the summary behind
            analysis/concordance/b4_pair_level_intermediacy.py
            <- same inputs
  Table S8  expert-review ledger, item ids and domain labels only:
            a) the 16 discordant Bayley-4 items, b) the 60 discordant Bayley-III
            items, each with Expert 1, Expert 2 and consensus
            <- expert_review/records/coauthor_review_b4.xlsx and
               coauthor_review_b3_solution.xlsx (NOT committed: they reproduce the
               item instructions, docs/DATA_ACCESS.md) + the mismatch ledgers.
               Regenerated only when the workbooks are present; otherwise the
               committed CSVs are kept and a message is printed.

All tables are written to results/tables/ as CSV.  Item titles come from
results/tables/bayley3_item_domain_assignments.csv (analysis/item_assignments.py),
so the script does not need the proprietary item text.

Usage (from the repository root):
    python tables/make_supplementary_tables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
PSY = ROOT / "psychometrics"

FACTOR_LABEL = {"GDP": "GDPS", "COG": "COG (unidimensional)"}
ABBREV = {"attention": "ATT", "working_memory": "WM", "goal_directed_problem_solving": "GDPS",
          "flexibility_shift": "FS", "higher_order_processing": "HOP"}
DETECTED = {"attention": "ATT", "working memory": "WM", "goal directed problem solving": "GDPS",
            "flexibility shift": "FS", "higher order processes": "HOP", "higher order processing": "HOP"}
DOMAIN_ORDER = ["ATT", "WM", "GDPS", "FS", "HOP"]
REF_YAML = ROOT / "config/reference_models_b4.yaml"
STEP2_CMP = (ROOT / "data/outputs/step2_b4_validation"
             / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
             / "05_model_comparison/item_comparison_theoretical.csv")
NONE = "\u2014"      # em dash: blank expert answer / no consensus / no expert domain
DOMINANCE_THRESHOLD = 90.0            # >= this share of identical responses = floor/ceiling
ADMINISTERED = range(34, 69)          # items 34-68 = the dHCP administration range


def item_titles() -> dict[int, str]:
    a = pd.read_csv(OUT / "bayley3_item_domain_assignments.csv")
    return dict(zip(a["item_number"], a["item_title"]))


def cfa_items() -> set[int]:
    inp = (PSY / "measurement/model_cfa_3factor.inp").read_text(encoding="utf-8", errors="ignore")
    use = re.search(r"USEVARIABLES ARE(.*?);", inp, flags=re.S).group(1)
    return {int(m) for m in re.findall(r"COG(\d{3})", use)}


# ---------------------------------------------------------------------------
# Table S1
# ---------------------------------------------------------------------------
def aylward_domains() -> dict[str, list[str]]:
    """item -> Aylward domain abbreviations, first-listed first (yaml order)."""
    import yaml
    ref = yaml.safe_load(REF_YAML.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for f in ABBREV:
        for i in ref["theoretical"].get(f, []):
            out.setdefault(i, []).append(ABBREV[f])
    return out


def table_s1() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/outputs/step1_translation/b4_b3_cascade_pairs.csv")
    df = df.sort_values(["b4_factor", "b4_item_id"]).reset_index(drop=True)
    # Three items carry two Aylward domains (COG_035 GDPS+FS, COG_048 GDPS+HOP,
    # COG_051 FS+HOP); the cascade transferred them under the first-listed one.
    ayl = aylward_domains()
    assert all(ayl[i][0] == ABBREV[f] for i, f in zip(df.b4_item_id, df.b4_factor)), \
        "b4_factor must equal the first-listed Aylward domain"
    s1 = pd.DataFrame({
        "b4_item_id": df.b4_item_id,
        "b4_item_title": df.b4_item_title,
        "b3_item_id": df.b3_item_id,
        "b3_item_title": df.b3_item_title,
        "aylward_domains": [" + ".join(ayl[i]) for i in df.b4_item_id],
        "theoretical_domain": df.b4_factor,
        "cosine_similarity_fulltext": df.sim_fulltext.round(3),
        "cosine_similarity_title": df.sim_title.round(3),
        "pairing_stage": df.pairing_stage.map({"title_locked": "title-locked",
                                               "fulltext": "full-text fallback"}),
    })
    s1.to_csv(OUT / "table_s1_cascade_pairs.csv", index=False)
    sim = s1.cosine_similarity_fulltext
    print(f"Table S1: {len(s1)} rows; full-text cosine mean={sim.mean():.3f} median={sim.median():.3f} "
          f"range=[{sim.min():.3f}, {sim.max():.3f}]; stages={s1.pairing_stage.value_counts().to_dict()}")
    return s1


# ---------------------------------------------------------------------------
# Table S2
# ---------------------------------------------------------------------------
def table_s2() -> pd.DataFrame:
    varnames = [l.strip() for l in (PSY / "varnames.txt").read_text().splitlines() if l.strip()]
    data = np.loadtxt(PSY / "combined.dat")
    fmap = pd.read_csv(PSY / "factor_item_map.csv")
    factor_of = dict(zip(fmap["item_num"].astype(int), fmap["factor"].replace({"GDP": "GDPS"})))
    titles = item_titles()
    assignments = pd.read_csv(OUT / "bayley3_item_domain_assignments.csv").set_index("item_number")
    retained_in_cfa = cfa_items()

    rows = []
    for num in ADMINISTERED:
        item = f"COG{num:03d}"
        row = {"item_id": item, "item_title": titles.get(num, ""),
               "translated_domain": assignments.loc[num, "domain"],
               "empirical_factor": factor_of.get(num, "")}
        if item in varnames:
            col = data[:, varnames.index(item)]
            valid = col[col != -999]
            n_valid = len(valid)
            pct1 = 100 * int((valid == 1).sum()) / n_valid
            pct0 = 100 * int((valid == 0).sum()) / n_valid
            dominant = max(pct1, pct0)
            status = ("ceiling" if pct1 >= pct0 else "floor") if dominant >= DOMINANCE_THRESHOLD else "retained"
            row |= {"n_valid": n_valid, "pct_pass": round(pct1, 1), "pct_fail": round(pct0, 1),
                    "dominant_response_pct": round(dominant, 1), "status": status}
        else:
            # Items whose domain had fewer than two indicators in the window
            # (COG059 is the only attention item) are not exported to combined.dat.
            row |= {"n_valid": "", "pct_pass": "", "pct_fail": "", "dominant_response_pct": "",
                    "status": "excluded: domain with a single indicator"}
        rows.append(row)
    s2 = pd.DataFrame(rows)
    retained = {int(r.item_id[3:]) for r in s2.itertuples() if r.status == "retained"}
    if retained != retained_in_cfa:
        raise SystemExit("Table S2 status disagrees with model_cfa_3factor.inp: "
                         f"only in table {sorted(retained - retained_in_cfa)}, "
                         f"only in model {sorted(retained_in_cfa - retained)}")
    s2.to_csv(OUT / "table_s2_administered_items.csv", index=False)
    print(f"Table S2: {len(s2)} rows; status counts {s2.status.value_counts().to_dict()}; "
          f"retained set = the 23 CFA items")
    return s2


# ---------------------------------------------------------------------------
# Table S3 -- Mplus parsers
# ---------------------------------------------------------------------------
def _stdyx_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = text.index("STDYX Standardization")
    end = text.index("STDY Standardization", start)
    return text[start:end]


ROW = re.compile(r"^\s*(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_loadings(path: Path, factors: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    loadings, corrs = [], []
    mode = cur = None
    for line in _stdyx_block(path).splitlines():
        st = line.strip()
        if re.match(r"^(Means|Intercepts|Thresholds|Variances|Residual Variances|Scales)\b", st):
            mode = None
            continue
        m = re.match(r"^([A-Z]{2,4})\s+(BY|WITH)\s*$", st)
        if m and m.group(1) in factors:
            cur, mode = m.groups()
            continue
        r = ROW.match(line)
        if not (r and mode):
            continue
        name, est, se, z, p = r.group(1), *map(float, r.groups()[1:])
        if mode == "BY" and name.startswith("COG"):
            loadings.append({"factor": FACTOR_LABEL.get(cur, cur), "item_id": name,
                             "std_loading": est, "se": se, "est_se": z, "p_value": p})
        elif mode == "WITH" and name in factors:
            corrs.append({"factor_pair": f"{FACTOR_LABEL.get(cur, cur)}-{FACTOR_LABEL.get(name, name)}",
                          "std_correlation": est, "se": se, "est_se": z, "p_value": p})
    return loadings, corrs


def parse_fit(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def grab(anchor: str, key: str, window: int = 8):
        for i, l in enumerate(lines):
            if anchor in l:
                for l2 in lines[i:i + window]:
                    m = re.search(rf"{key}\s+(-?[0-9.]+)", l2)
                    if m:
                        return float(m.group(1))
        return None

    fit = {
        "chi_square": grab("Chi-Square Test of Model Fit", "Value"),
        "df": grab("Chi-Square Test of Model Fit", "Degrees of Freedom"),
        "chi_p": grab("Chi-Square Test of Model Fit", "P-Value"),
        "rmsea": grab("RMSEA (Root", "Estimate"),
        "cfi": grab("CFI/TLI", "CFI"),
        "tli": grab("CFI/TLI", "TLI"),
        "srmr": grab("SRMR (Standardized", "Value"),
    }
    # 90 % CI of the RMSEA is printed as "90 Percent C.I.   0.055  0.064"
    for i, l in enumerate(lines):
        if "RMSEA (Root" in l:
            for l2 in lines[i:i + 8]:
                m = re.search(r"90 Percent C\.I\.\s+([0-9.]+)\s+([0-9.]+)", l2)
                if m:
                    fit["rmsea_ci_low"], fit["rmsea_ci_high"] = float(m.group(1)), float(m.group(2))
            break
    fit["df"] = int(fit["df"]) if fit["df"] is not None else None
    return fit


def parse_difftest(path: Path) -> dict:
    """The 'Chi-Square Test for Difference Testing' block of an Mplus .out."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, l in enumerate(lines):
        if "Chi-Square Test for Difference Testing" in l:
            block = "\n".join(lines[i:i + 8])
            chi = re.search(r"Value\s+([0-9.]+)", block)
            df = re.search(r"Degrees of Freedom\s+(\d+)", block)
            pv = re.search(r"P-Value\s+([0-9.]+)", block)
            if chi and df and pv:
                return {"chi2": float(chi.group(1)), "df": int(df.group(1)), "p": float(pv.group(1))}
    raise SystemExit(f"no DIFFTEST block in {path}")


def table_s3() -> None:
    titles = item_titles()
    three = PSY / "measurement/model_cfa_3factor.out"
    uni = PSY / "measurement/model_unidimensional.out"

    loadings, corrs = parse_loadings(three, ("WM", "GDP", "FS"))
    s3a = pd.DataFrame(loadings)
    s3a.insert(2, "item_title", s3a["item_id"].map(lambda x: titles.get(int(x[3:]), "")))
    s3a.to_csv(OUT / "table_s3a_cfa_3factor_loadings.csv", index=False)
    s3b = pd.DataFrame(corrs)
    s3b.to_csv(OUT / "table_s3b_cfa_3factor_correlations.csv", index=False)

    fit3, fit1 = parse_fit(three), parse_fit(uni)
    # Nested-model comparison: the unidimensional model (all three factor
    # correlations fixed to 1) tested against the three-factor model with the
    # WLSMV-corrected chi-square difference test (Mplus DIFFTEST, 3 df).
    diff = parse_difftest(PSY / "measurement/model_unidimensional_difftest.out")
    s3c = pd.DataFrame([
        {"model": "3-factor (WM, GDPS, FS)", "n_items": len(s3a), **fit3,
         "difftest_vs_3factor_chi2": None, "difftest_df": None, "difftest_p": None,
         "source": "psychometrics/measurement/model_cfa_3factor.out"},
        {"model": "Unidimensional (COG)", "n_items": len(s3a), **fit1,
         "difftest_vs_3factor_chi2": diff["chi2"], "difftest_df": diff["df"], "difftest_p": diff["p"],
         "source": "psychometrics/measurement/model_unidimensional.out; difference test: "
                   "model_unidimensional_difftest.out (H0) vs model_cfa_3factor_difftest.out (H1)"},
    ])
    s3c["difftest_df"] = s3c["difftest_df"].astype("Int64")
    s3c.to_csv(OUT / "table_s3c_cfa_fit_statistics.csv", index=False)

    uni_loadings, _ = parse_loadings(uni, ("COG",))
    s3d = pd.DataFrame(uni_loadings).drop(columns=["factor"])
    s3d.insert(1, "item_title", s3d["item_id"].map(lambda x: titles.get(int(x[3:]), "")))
    s3d.to_csv(OUT / "table_s3d_cfa_unidimensional_loadings.csv", index=False)

    print(f"Table S3a: {len(s3a)} loadings (range {s3a.std_loading.min():.2f}-{s3a.std_loading.max():.2f}); "
          f"S3b: {len(s3b)} correlations {s3b.set_index('factor_pair').std_correlation.round(2).to_dict()}")
    print("Table S3c: 3-factor RMSEA/CFI/TLI/SRMR = "
          f"{fit3['rmsea']:.3f}/{fit3['cfi']:.3f}/{fit3['tli']:.3f}/{fit3['srmr']:.3f}; unidimensional = "
          f"{fit1['rmsea']:.3f}/{fit1['cfi']:.3f}/{fit1['tli']:.3f}/{fit1['srmr']:.3f}")
    print(f"           nested-model DIFFTEST unidimensional vs 3-factor: chi2({diff['df']}) = {diff['chi2']:.3f}, "
          f"p = {diff['p']:.4f}")
    print(f"Table S3d: {len(s3d)} unidimensional loadings")


# ---------------------------------------------------------------------------
# Table S4 / S5
# ---------------------------------------------------------------------------
def table_s4() -> None:
    s4 = pd.read_csv(PSY / "mimic/mimic_std_effects.csv")
    s4["factor"] = s4["factor"].replace(FACTOR_LABEL)
    s4 = s4.rename(columns={"subfolder": "covariate", "label": "covariate_label"})
    s4.to_csv(OUT / "table_s4a_mimic_covariate_associations.csv", index=False)
    diff = pd.read_csv(PSY / "mimic/difftest_results.csv")
    diff = diff.rename(columns={"predictor": "covariate"})
    diff["constraint"] = diff["constraint"].str.replace("GDP", "GDPS", regex=False)
    diff.to_csv(OUT / "table_s4b_mimic_difftest.csv", index=False)
    print(f"Table S4a: {len(s4)} rows, {s4['covariate'].nunique()} covariates "
          f"(standardisation: {s4['standardization'].value_counts().to_dict()}); "
          f"S4b: {len(diff)} DIFFTEST contrasts, {int(diff['significant'].sum())} significant")


def table_s5() -> None:
    s5 = pd.read_csv(PSY / "outcomes/outcomes_associations.csv").drop_duplicates()
    s5["factor"] = s5["factor"].replace(FACTOR_LABEL)
    s5 = s5.rename(columns={"subfolder": "outcome_code", "label": "outcome_label"})
    s5.to_csv(OUT / "table_s5_factor_outcome_correlations.csv", index=False)
    print(f"Table S5: {len(s5)} rows, {s5['outcome_code'].nunique()} outcomes")


# ---------------------------------------------------------------------------
# Table S6 / S7 -- Bayley-4 clusters vs the Aylward reference models
# ---------------------------------------------------------------------------
def _b4_reference_and_clusters():
    import yaml
    ref = yaml.safe_load(REF_YAML.read_text(encoding="utf-8"))
    theo_all = aylward_domains()
    theo_prim = {i: v[0] for i, v in theo_all.items()}
    emp = {i: f for f, items in ref["empirical"].items()
           if f not in ("description", "source") for i in items}
    cmp = pd.read_csv(STEP2_CMP)
    det = {r.item_id: DETECTED[str(r.detected_factor).strip().lower()] for r in cmp.itertuples()}
    return theo_all, theo_prim, emp, det


def table_s6() -> pd.DataFrame:
    theo_all, theo_prim, _, det = _b4_reference_and_clusters()
    multi = {i for i, v in theo_all.items() if len(v) > 1}
    rows = []
    for d in DOMAIN_ORDER:
        any_items = [i for i, v in theo_all.items() if d in v]
        single = [i for i in any_items if i not in multi]
        rows.append({"domain": d,
                     "items_any_factor": len(any_items),
                     "recovered_any_factor": sum(det[i] == d for i in any_items),
                     "pct_any_factor": round(100 * sum(det[i] == d for i in any_items) / len(any_items), 1),
                     "items_single_domain": len(single),
                     "recovered_single_domain": sum(det[i] == d for i in single),
                     "pct_single_domain": round(100 * sum(det[i] == d for i in single) / len(single), 1)})
    # totals count each ITEM once: recovered if its cluster matches any listed domain
    any_tot = sum(det[i] in v for i, v in theo_all.items())
    single_tot = sum(det[i] == theo_prim[i] for i in theo_all if i not in multi)
    n_single = len(theo_all) - len(multi)
    rows.append({"domain": "All items",
                 "items_any_factor": len(theo_all), "recovered_any_factor": any_tot,
                 "pct_any_factor": round(100 * any_tot / len(theo_all), 1),
                 "items_single_domain": n_single, "recovered_single_domain": single_tot,
                 "pct_single_domain": round(100 * single_tot / n_single, 1)})
    s6 = pd.DataFrame(rows)
    s6.to_csv(OUT / "table_s6_b4_domain_recovery.csv", index=False)
    tot = s6.iloc[-1]
    print(f"Table S6: {len(s6)} rows; any-factor recovery "
          f"{int(tot.recovered_any_factor)}/{int(tot.items_any_factor)} ({tot.pct_any_factor}%), per domain "
          + ", ".join(f"{r.domain} {r.recovered_any_factor}/{r.items_any_factor}" for r in s6.itertuples() if r.domain != "All items"))
    return s6


def table_s7() -> pd.DataFrame:
    import itertools
    theo_all, theo_prim, emp, det = _b4_reference_and_clusters()
    multi = {i for i, v in theo_all.items() if len(v) > 1}
    common = sorted(i for i in det if i in emp and i in theo_all)     # 34 items
    strict = [i for i in common if i not in multi]                     # 31 items
    treatments = {
        "single_domain": (strict, lambda a, b: theo_prim[a] == theo_prim[b]),
        "any_factor": (common, lambda a, b: bool(set(theo_all[a]) & set(theo_all[b]))),
        "first_listed": (common, lambda a, b: theo_prim[a] == theo_prim[b]),
    }

    def stats(items, together):
        df = pd.DataFrame([{"theo": bool(together(a, b)), "emp": emp[a] == emp[b], "det": det[a] == det[b]}
                           for a, b in itertools.combinations(items, 2)])
        ct, ca, dis = df[df.theo & df.emp], df[~df.theo & ~df.emp], df[df.theo != df.emp]
        d_theo, d_emp = dis[dis.theo], dis[dis.emp]
        return {
            ("Consensus-apart pairs", "Pairs"): len(ca),
            ("Consensus-apart pairs", "Kept apart by the clusters"): f"{int((~ca.det).sum())} ({100 * (~ca.det).mean():.0f}%)",
            ("Consensus-together pairs", "Pairs"): len(ct),
            ("Consensus-together pairs", "Kept together by the clusters"): f"{int(ct.det.sum())} ({100 * ct.det.mean():.0f}%)",
            ("Disputed pairs", "Pairs"): len(dis),
            ("Disputed pairs", "Resolved toward the expert model"): int(d_theo.det.sum()) + int((~d_emp.det).sum()),
            ("Disputed pairs", "Resolved toward the empirical model"): int((~d_theo.det).sum()) + int(d_emp.det.sum()),
            ("Pairs placed together by each model", "Expert model"): int(df.theo.sum()),
            ("Pairs placed together by each model", "Empirical model"): int(df.emp.sum()),
            ("Pairs placed together by each model", "Detected clusters"): int(df.det.sum()),
        }

    res = {name: stats(items, fn) for name, (items, fn) in treatments.items()}
    keys = list(res["single_domain"])
    s7 = pd.DataFrame([{"pair_class": k[0], "quantity": k[1],
                        **{f"{n}_{len(treatments[n][0])}_items": res[n][k] for n in treatments}} for k in keys])
    s7.to_csv(OUT / "table_s7_b4_pair_level_intermediacy.csv", index=False)
    ss = res["single_domain"]
    print(f"Table S7: {len(s7)} rows; 31 single-domain items: consensus-apart kept apart "
          f"{ss[keys[1]]}, consensus-together kept {ss[keys[3]]}, disputed {ss[keys[4]]} -> expert "
          f"{ss[keys[5]]} / empirical {ss[keys[6]]}; together-pairs expert/empirical/detected "
          f"{ss[keys[7]]}/{ss[keys[8]]}/{ss[keys[9]]}")
    return s7


# ---------------------------------------------------------------------------
# Table S8 -- expert-review ledger (needs the uncommitted review workbooks)
# ---------------------------------------------------------------------------
def table_s8() -> None:
    rec = ROOT / "expert_review/records"
    wb4, wb3 = rec / "coauthor_review_b4.xlsx", rec / "coauthor_review_b3_solution.xlsx"
    out_a, out_b = OUT / "table_s8a_expert_review_b4.csv", OUT / "table_s8b_expert_review_b3.csv"
    if not (wb4.exists() and wb3.exists()):
        print(f"Table S8: review workbooks not present ({wb4.name}, {wb3.name}; docs/DATA_ACCESS.md) -- "
              f"keeping the committed {out_a.name} / {out_b.name}")
        return
    import json
    sys.path.insert(0, str(ROOT / "expert_review"))
    import import_review_b3 as imp3
    import import_review_b4 as imp4

    def ab(x):
        return NONE if x is None or pd.isna(x) else ABBREV[x]

    def sort_by(df, col):
        rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
        return (df.assign(_k=df[col].map(rank).fillna(len(rank))).sort_values(["_k", "item_id"])
                  .drop(columns="_k").reset_index(drop=True))

    # -- S8a: Bayley-4 --------------------------------------------------------
    rev4 = imp4.load_b4_review(wb4)
    led4 = pd.read_csv(OUT / "mismatches_ledger_b4.csv")
    kap4 = json.loads((rec / "coauthor_review_b4_kappa.json").read_text(encoding="utf-8"))
    rev4 = rev4[~rev4.item_id.isin(kap4["items_unrated"])]
    assert set(rev4.item_id) == set(led4.item_id), "Bayley-4 review set != mismatches ledger"
    s8a = (led4[["item_id", "theoretical_factors", "detected_abbrev"]]
           .rename(columns={"theoretical_factors": "aylward_domain", "detected_abbrev": "detected_cluster"})
           .merge(rev4[["item_id", "expert1", "expert2", "consensus"]], on="item_id"))
    s8a["aylward_domain"] = s8a.aylward_domain.replace({"-": NONE})
    for c in ("expert1", "expert2", "consensus"):
        s8a[c] = s8a[c].map(ab)
    s8a = sort_by(s8a, "aylward_domain").rename(columns={"expert1": "expert_1", "expert2": "expert_2"})
    assert len(s8a) == 16 and int((s8a.consensus != NONE).sum()) == kap4["n_consensus"]
    s8a.to_csv(out_a, index=False)

    # -- S8b: Bayley-III ------------------------------------------------------
    items = imp3.load_items()
    rev3 = imp3.load_review_xlsx(wb3, items)
    m = rev3.merge(imp3.load_step1_assignments(), on="item_id", how="left") \
            .merge(imp3.load_step3_assignments(), on="item_id", how="left")
    m["final_factor"] = [c if (uc and not pd.isna(c)) else s1
                         for uc, c, s1 in zip(m.is_uc, m.consensus, m.step1_factor)]
    disc = m[m.final_factor != m.step3_factor].copy()
    led3 = pd.read_csv(OUT / "mismatches_ledger_b3.csv")
    assert set(disc.item_id) == set(led3.item_id) and len(disc) == 60, "Bayley-III discordant set != ledger"
    final = pd.read_csv(OUT / "bayley3_item_domain_assignments.csv").set_index("item_id")
    for r in disc.itertuples():
        assert ABBREV[r.step1_factor] == final.loc[r.item_id, "algorithmic_domain"], r.item_id
        assert ABBREV[r.final_factor] == final.loc[r.item_id, "domain"], r.item_id
    s8b = pd.DataFrame({"item_id": disc.item_id,
                        "algorithmic_label": disc.step1_factor.map(ab),
                        "final_label": disc.final_factor.map(ab),
                        "bottom_up_cluster": disc.step3_factor.map(ab),
                        "expert_1": disc.expert1.map(ab), "expert_2": disc.expert2.map(ab),
                        "consensus": disc.consensus.map(ab)})
    s8b = sort_by(s8b, "final_label")
    assert int((s8b.algorithmic_label != s8b.final_label).sum()) == 3        # the three overrides
    s8b.to_csv(out_b, index=False)
    print(f"Table S8a: {len(s8a)} discordant Bayley-4 items ({int((s8a.consensus != NONE).sum())} with consensus); "
          f"S8b: {len(s8b)} discordant Bayley-III items ({int((s8b.consensus != NONE).sum())} with consensus, "
          f"3 overrides)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table_s1()
    table_s2()
    table_s3()
    table_s4()
    table_s5()
    table_s6()
    table_s7()
    table_s8()
    print(f"\nAll supplementary tables written to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
