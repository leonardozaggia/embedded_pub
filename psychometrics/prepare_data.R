# ============================================================================
# Mplus Data Preparation
# ============================================================================
# Exports a single space-delimited combined.dat with:
#   - 34 binary cognitive items (2PL model, items 34-68 excl. 059)
#     COG059 ("Attends to Story") is the sole item on the attention factor
#     in the items 34-68 range; a single-indicator latent factor is
#     unidentifiable, so it is excluded.
#   - 23 perinatal/demographic predictors (MIMIC model; MomEth split into 4 dummies;
#       EPDSTot, StimEnv, PrLax, PrOvr, PrVrb, PrSty added from epds01, stps01, pqmf01)
#   - 15 continuous outcome variables (structural model; QchatTot added from qucht01)
#
# Missing values coded as -999 (declared in Mplus with MISSING = -999).
# Inputs (restricted dHCP / NDA data, see docs/DATA_ACCESS.md):
#   data/processed/cogn_id_GA.xlsx   item-level Bayley-III cognition responses
#   data/raw/dhcp_txt/*.txt          NDA instruments (lpb01, cpenr01, cbcl1_501, ...)
# Outputs (psychometrics/):
#   combined.dat, varnames.txt, factor_item_map.csv
# Run before any Mplus model:  Rscript psychometrics/prepare_data.R
# ============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
})

# ── paths ────────────────────────────────────────────────────────────────────
# The repository root is derived from this script's own location, so the script
# can be run from any working directory:  Rscript psychometrics/prepare_data.R
script_path <- {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) normalizePath(sub("^--file=", "", m[1])) else normalizePath("psychometrics/prepare_data.R")
}
base_path    <- dirname(dirname(script_path))
dhcp_txt_dir <- file.path(base_path, "data/raw/dhcp_txt")
cogn_file    <- file.path(base_path, "data/processed/cogn_id_GA.xlsx")
lpb_file     <- file.path(dhcp_txt_dir, "lpb01.txt")
demo_file    <- file.path(dhcp_txt_dir, "cpenr01.txt")
cbcl_file    <- file.path(dhcp_txt_dir, "cbcl1_501.txt")
ecbq_file    <- file.path(dhcp_txt_dir, "ecbq01.txt")
bsid_file    <- file.path(dhcp_txt_dir, "bsid_iii01.txt")
qucht_file   <- file.path(dhcp_txt_dir, "qucht01.txt")
stps_file    <- file.path(dhcp_txt_dir, "stps01.txt")
pqmf_file    <- file.path(dhcp_txt_dir, "pqmf01.txt")
epds_file    <- file.path(dhcp_txt_dir, "epds01.txt")

out_dir <- file.path(base_path, "psychometrics")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ── item specification ───────────────────────────────────────────────────────
# Items 34-68 grouped by the translated domain structure (Objective 1; step-1
# output config/model_specs.yaml, key
# all_mpnet_base_v2_b3_expert_guided_v6_cog_mixed_centroids, i.e. the
# mixed_centroids solution).  The floor/ceiling screen that reduces these 34
# items to the 23 CFA indicators is applied in the Mplus .inp files
# (USEVARIABLES), not here -- see results/tables/table_s2_administered_items.csv.
# Items 34-68; COG059 excluded because it is the only attention-factor item
# in this range.  A latent factor with a single indicator is unidentifiable
# in a confirmatory IRT/SEM model, so the item (and its factor) are dropped.
EXCLUDED_ITEMS <- c(59L)

# Factor structure from b3_model_spec.yaml (mixed_centroids)
# COG068 in WM; COG047 in FS
FACTOR_ITEMS <- list(
  WM  = c(34, 40, 45, 50, 64, 68),
  GDP = c(37, 38, 39, 41, 43, 44, 46, 48, 52, 53, 57, 65, 67),
  FS  = c(35, 36, 42, 47, 49, 51, 54, 55, 56, 58, 60, 61, 62, 63, 66)
)
all_model_items <- sort(unlist(FACTOR_ITEMS, use.names = FALSE))  # 32 items

# ── helpers ──────────────────────────────────────────────────────────────────
read_ndar <- function(path) {
  # NDAR tab-delimited: row 1 = variable names, row 2 = value ranges (drop it)
  df <- read.delim(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
  df[-1, ]  # drop value-range row
}

to_num <- function(x) {
  # Convert to numeric and replace NDAR missing sentinels with NA.
  # NDAR uses -999 as the primary missing code; some instruments (e.g. CBCL)
  # also use 999. Both are replaced here so downstream logic is correct.
  # NAs are converted back to -999 at export time for Mplus.
  n <- suppressWarnings(as.numeric(x))
  n[!is.na(n) & (n %in% c(-999L, -998L, 999L))] <- NA_real_
  n
}

dedup_ndar <- function(df) {
  # One row per subject: keep the earliest assessment (lowest interview_age).
  # Handles multi-visit instruments (e.g. EPDS given at neonatal and 18-month
  # visits). For single-visit instruments this is a no-op safety step.
  df |>
    arrange(src_subject_id, interview_age) |>
    distinct(src_subject_id, .keep_all = TRUE)
}

# ── load data ─────────────────────────────────────────────────────────────────
cat("Loading cognitive items...\n")
df_cogn <- read_excel(cogn_file)

# Guard: scan-based GA column may not exist in all versions of the file
if (!"nscan_ga_at_birth_weeks" %in% names(df_cogn)) {
  df_cogn$nscan_ga_at_birth_weeks <- NA_real_
  message("nscan_ga_at_birth_weeks not found in cogn_id_GA.xlsx; GA will fall back to baby_gest_at_birth")
}

# Select items in range 34-68, excluding sparse-factor items
cog_cols_all <- grep("^bsid_cog\\d+$", names(df_cogn), value = TRUE)
cog_nums     <- as.integer(sub("^bsid_cog", "", cog_cols_all))
cog_cols     <- cog_cols_all[
  !is.na(cog_nums) &
  cog_nums %in% all_model_items
]
cog_cols <- cog_cols[order(as.integer(sub("^bsid_cog", "", cog_cols)))]
cat(sprintf("  %d model items selected\n", length(cog_cols)))

cat("Loading perinatal file (lpb01)...\n")
df_lpb <- read_ndar(lpb_file) |>
  mutate(across(-src_subject_id, to_num))

cat("Loading demographics file (cpenr01)...\n")
df_demo <- read_ndar(demo_file) |>
  mutate(across(-src_subject_id, to_num))

cat("Loading CBCL file (cbcl1_501)...\n")
df_cbcl <- read_ndar(cbcl_file) |>
  mutate(across(-src_subject_id, to_num))

cat("Loading ECBQ file (ecbq01)...\n")
df_ecbq <- read_ndar(ecbq_file) |>
  mutate(across(-src_subject_id, to_num))

cat("Loading BSID-III file (bsid_iii01) for motor/language composites...\n")
df_bsid <- read_ndar(bsid_file) |>
  mutate(across(-src_subject_id, to_num))

cat("Loading QCHAT file (qucht01) for autism screening total...\n")
df_qucht <- read_ndar(qucht_file) |>
  mutate(across(-src_subject_id, to_num)) |>
  dedup_ndar()

cat("Loading STPS file (stps01) for home stimulation environment...\n")
df_stps <- read_ndar(stps_file) |>
  mutate(across(-src_subject_id, to_num)) |>
  dedup_ndar()

cat("Loading PQMF file (pqmf01) for parenting style scales...\n")
df_pqmf <- read_ndar(pqmf_file) |>
  mutate(across(-src_subject_id, to_num)) |>
  dedup_ndar()

cat("Loading EPDS file (epds01) for maternal depression total...\n")
df_epds <- read_ndar(epds_file) |>
  mutate(across(-src_subject_id, to_num)) |>
  dedup_ndar()

# ── merge ─────────────────────────────────────────────────────────────────────
cat("Merging datasets...\n")

# Items: use src_subject_id from cogn_id_GA.xlsx
item_df <- df_cogn |>
  select(src_subject_id, nscan_ga_at_birth_weeks, all_of(cog_cols)) |>
  mutate(across(all_of(cog_cols), to_num))

master <- item_df |>
  left_join(
    df_lpb |> select(src_subject_id,
      baby_gest_at_birth, baby_birth_weight, baby_gender,
      baby_babyhc, baby_suspected_iugr, study_gestational_diabetes,
      study_preeclampsia, study_corticosteroids),
    by = "src_subject_id"
  ) |>
  left_join(
    df_demo |> select(src_subject_id,
      mother_age1, mother_education1, father_education1,
      mother_ethnicity1, mother_eng_first_lan),
    by = "src_subject_id"
  ) |>
  left_join(
    df_cbcl |> select(src_subject_id,
      cbcl_internal_raw, cbcl_external_raw, cbcl_total_raw,
      cbcl_anxious_raw, cbcl_withdrawn_raw, cbcl_somatic_c_raw,
      cbcl_sleep_raw, cbcl_attention_raw, cbcl_aggressive_raw,
      cbcl_emotional_raw),
    by = "src_subject_id"
  ) |>
  left_join(
    df_ecbq |> select(src_subject_id,
      ecbq_surgency, ecbq_negative_aff, ecbq_effort_cntrl),
    by = "src_subject_id"
  ) |>
  left_join(
    # NOTE: if bsid_iii01 uses 'subjectkey' instead of 'src_subject_id',
    # change the join key accordingly.
    df_bsid |> select(src_subject_id,
      bsid_mot_composite, bsid_lang_composite),
    by = "src_subject_id"
  ) |>
  left_join(
    df_qucht |> select(src_subject_id, qchat_total),
    by = "src_subject_id"
  ) |>
  left_join(
    df_stps |> select(src_subject_id, total_sps_home_environment),
    by = "src_subject_id"
  ) |>
  left_join(
    df_pqmf |> select(src_subject_id,
      pm_laxness, pm_overactivity, pm_verbosity, pm_parent_style),
    by = "src_subject_id"
  ) |>
  left_join(
    df_epds |> select(src_subject_id, epds_tot),
    by = "src_subject_id"
  )

# ── recode & rename ───────────────────────────────────────────────────────────
# GA: primary source is baby_gest_at_birth from lpb01 (highest coverage,
# directly reported gestational age in weeks).  Fall back to the scan-derived
# nscan_ga_at_birth_weeks from cogn_id_GA.xlsx only when lpb01 is missing.
master <- master |>
  mutate(
    GA      = ifelse(!is.na(to_num(baby_gest_at_birth)),
                     to_num(baby_gest_at_birth),
                     to_num(nscan_ga_at_birth_weeks)),
    BirthWt = to_num(baby_birth_weight),
    # Sex: recode so 1 = Male, 0 = Female (baby_gender: 1=M, 2=F)
    Sex     = case_when(to_num(baby_gender) == 1 ~ 1,
                        to_num(baby_gender) == 2 ~ 0,
                        TRUE ~ NA_real_),
    # HdCirc: clip values outside ±4 SD to NA (biologically implausible entries)
    HdCirc  = {
      hc      <- to_num(baby_babyhc)
      hc_mu   <- mean(hc, na.rm = TRUE)
      hc_sd   <- sd(hc, na.rm = TRUE)
      hc[!is.na(hc) & abs(hc - hc_mu) > 4 * hc_sd] <- NA_real_
      hc
    },
    # IUGR: 0 = No, 1 = Yes (suspected IUGR)
    IUGR    = to_num(baby_suspected_iugr),
    # GDiab: 0 = No, 1 = Yes (gestational diabetes)
    GDiab   = to_num(study_gestational_diabetes),
    # PrEcl dichotomised: 0 = no hypertensive disorder, 1 = any (codes 1/2/3)
    PrEcl   = case_when(to_num(study_preeclampsia) == 0             ~ 0,
                        to_num(study_preeclampsia) %in% c(1, 2, 3) ~ 1,
                        TRUE ~ NA_real_),
    # AntCor: dummy-coded from study_corticosteroids (ref = 0 = No corticosteroids)
    #   AntCor_P = 1 if Partial course  (code 1, n ~ 36),  0 if None or Complete
    #   AntCor_C = 1 if Complete course (code 2, n ~ 121), 0 if None or Partial
    #   NA propagated for missing values.
    .antcor   = to_num(study_corticosteroids),
    AntCor_P  = ifelse(!is.na(.antcor), as.integer(.antcor == 1), NA_integer_),
    AntCor_C  = ifelse(!is.na(.antcor), as.integer(.antcor == 2), NA_integer_),
    MomAge  = to_num(mother_age1),
    MomEdu  = to_num(mother_education1),
    DadEdu  = to_num(father_education1),
    # MomEth: 4 dummy variables (reference = White, ONS codes 1-4).
    # Code 18 (prefer not to say) -> NA on all dummies.
    .eth    = to_num(mother_ethnicity1),
    .eth_ok = !is.na(.eth) & .eth != 18,
    EthSAsian = ifelse(.eth_ok, as.integer(.eth %in% 5:8),              NA_integer_),
    EthBlack  = ifelse(.eth_ok, as.integer(.eth %in% c(10, 11, 15)),    NA_integer_),
    EthMixed  = ifelse(.eth_ok, as.integer(.eth %in% c(9, 12, 13, 16)), NA_integer_),
    EthOther  = ifelse(.eth_ok, as.integer(.eth %in% c(14, 17)),        NA_integer_),
    # MomEng: 0 = English NOT first language, 1 = English IS first language
    MomEng  = to_num(mother_eng_first_lan),
    # Outcomes
    CBInt   = to_num(cbcl_internal_raw),
    CBExt   = to_num(cbcl_external_raw),
    CBTot   = to_num(cbcl_total_raw),
    CBAnx   = to_num(cbcl_anxious_raw),
    CBWth   = to_num(cbcl_withdrawn_raw),
    CBSom   = to_num(cbcl_somatic_c_raw),
    CBSlp   = to_num(cbcl_sleep_raw),
    CBAtt   = to_num(cbcl_attention_raw),
    CBAgg   = to_num(cbcl_aggressive_raw),
    CBEmR   = to_num(cbcl_emotional_raw),
    ECSur   = to_num(ecbq_surgency),
    ECNeg   = to_num(ecbq_negative_aff),
    ECEff   = to_num(ecbq_effort_cntrl),
    BayMot  = to_num(bsid_mot_composite),
    BayLan  = to_num(bsid_lang_composite),
    # EPDS: maternal postnatal depression total (earliest available visit)
    EPDSTot = to_num(epds_tot),
    # QCHAT: autism screening total (18-month)
    QchatTot = to_num(qchat_total),
    # STPS: stimulating home environment composite
    StimEnv  = to_num(total_sps_home_environment),
    # PQMF parenting style subscales
    PrLax    = to_num(pm_laxness),
    PrOvr    = to_num(pm_overactivity),
    PrVrb    = to_num(pm_verbosity),
    PrSty    = to_num(pm_parent_style)
  ) |>
  select(-.eth, -.eth_ok, -.antcor)

# Rename item columns to Mplus-friendly names (COG034, COG035, ...)
item_mplus_names <- sprintf("COG%03d", as.integer(sub("^bsid_cog", "", cog_cols)))
names(master)[match(cog_cols, names(master))] <- item_mplus_names

# ── build final export frame ──────────────────────────────────────────────────
demo_vars    <- c("GA", "BirthWt", "Sex", "HdCirc", "IUGR",
                  "GDiab", "PrEcl", "AntCor_P", "AntCor_C",
                  "MomAge", "MomEdu", "DadEdu",
                  "EthSAsian", "EthBlack", "EthMixed", "EthOther",
                  "MomEng", "EPDSTot", "StimEnv", "PrLax", "PrOvr", "PrVrb",
                  "PrSty")

outcome_vars <- c("CBInt", "CBExt", "CBTot", "CBAnx", "CBWth",
                  "CBSom", "CBSlp", "CBAtt", "CBAgg",
                  "ECSur", "ECNeg", "ECEff", "BayMot", "BayLan",
                  "QchatTot", "CBEmR")

all_vars <- c(item_mplus_names, demo_vars, outcome_vars)

# Retain subjects with at least one item present; Mplus FIML handles
# missingness in covariates and outcomes (MLR estimator).
# Note: filter is applied before select so the row condition references
# the full master frame unambiguously.
keep_rows <- rowSums(!is.na(master[item_mplus_names])) > 0
export_df  <- master[keep_rows, all_vars]

cat(sprintf("Export sample: N = %d rows, %d variables\n",
            nrow(export_df), ncol(export_df)))

# Report missingness per block
report_miss <- function(df, vars, label) {
  n_complete <- sum(complete.cases(df[, vars]))
  cat(sprintf("  %s: %d / %d complete\n", label, n_complete, nrow(df)))
}
report_miss(export_df, item_mplus_names, "Items (all 32)")
report_miss(export_df, demo_vars,        "Demographics (all 23)")
report_miss(export_df, outcome_vars,     "Outcomes (all 15)")

# ── export ────────────────────────────────────────────────────────────────────
# Replace NA with -999 for Mplus (MISSING = -999 in .inp files)
export_mat <- export_df
export_mat[is.na(export_mat)] <- -999

out_file <- file.path(out_dir, "combined.dat")
write.table(
  export_mat,
  file      = out_file,
  sep       = " ",
  row.names = FALSE,
  col.names = FALSE,
  quote     = FALSE
)
cat(sprintf("Saved: %s\n", out_file))

# Variable names reference file (copy-paste into Mplus NAMES ARE)
varnames_file <- file.path(out_dir, "varnames.txt")
writeLines(all_vars, varnames_file)
cat(sprintf("Saved: %s  (%d variable names)\n", varnames_file, length(all_vars)))

# Factor map reference
factor_map <- data.frame(
  factor = rep(names(FACTOR_ITEMS), lengths(FACTOR_ITEMS)),
  item_num = unlist(FACTOR_ITEMS),
  mplus_name = sprintf("COG%03d", unlist(FACTOR_ITEMS)),
  r_name = sprintf("bsid_cog%d", unlist(FACTOR_ITEMS))
)
write.csv(factor_map, file.path(out_dir, "factor_item_map.csv"), row.names = FALSE)
cat(sprintf("Saved: factor_item_map.csv\n"))
