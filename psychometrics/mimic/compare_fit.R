# =============================================================================
# compare_fit.R  —  Model fit comparison and STDY effect extraction
# =============================================================================
# Purpose:
#   Compares fit indices (CFI, TLI, RMSEA, SRMR) across 3-factor and
#   unidimensional MIMIC models for all demographic predictors, and prints
#   STDY-standardised ON regression effects for each model type.
#
# Inputs (per predictor subfolder of this directory):
#   <PRED>/model_mimic_<PRED>.out         free 3-factor MIMIC output
#   <PRED>/model_mimic_<PRED>_unidim.out  unidimensional MIMIC output (if present)
#
# Intermediary files:  none
#
# Outputs:  printed to console only — no files written
#   - Fit index table (CFI, TLI, RMSEA, SRMR) by predictor x model
#   - STDY ON effects for 3-factor models
#   - STDY ON effects for unidimensional models
#
# Usage (from repo root):
#   Rscript psychometrics/mimic/compare_fit.R
# =============================================================================

library(dplyr)

# ── Parse STDY ON effects ─────────────────────────────────────────────────────
parse_stdy_effects <- function(path) {
  lines <- readLines(path, warn = FALSE)

  stdy_start <- grep("^STDY Standardization", lines)[1]
  if (is.na(stdy_start)) return(data.frame())

  next_section <- grep("^[A-Z][A-Z -]+$|^R-SQUARE", lines)
  next_section <- next_section[next_section > stdy_start][1]
  if (is.na(next_section)) next_section <- length(lines)

  section <- lines[stdy_start:next_section]

  rows <- list()
  current_factor <- NA_character_

  for (ln in section) {
    if (grepl("^\\s+[A-Z]+\\s+ON\\s*$", ln)) {
      current_factor <- trimws(sub("\\s+ON.*", "", ln))
      next
    }
    m <- regmatches(ln, regexec(
      "^\\s{4}(\\S+)\\s+(-?[0-9]+\\.[0-9]+)\\s+([0-9]+\\.[0-9]+)\\s+(-?[0-9]+\\.[0-9]+)\\s+([0-9]+\\.[0-9]+)",
      ln))[[1]]
    if (length(m) == 6 && !is.na(current_factor)) {
      rows[[length(rows)+1]] <- data.frame(
        Factor    = current_factor,
        Predictor = m[2],
        Estimate  = as.numeric(m[3]),
        SE        = as.numeric(m[4]),
        EstSE     = as.numeric(m[5]),
        Pvalue    = as.numeric(m[6]),
        stringsAsFactors = FALSE
      )
    }
    if (grepl("^\\s*$", ln) && !is.na(current_factor)) current_factor <- NA_character_
  }

  if (length(rows) == 0) return(data.frame())
  bind_rows(rows)
}

# ── Parse fit indices ─────────────────────────────────────────────────────────
parse_mplus_fit <- function(path) {
  lines <- readLines(path, warn = FALSE)

  get_section_val <- function(section_pat) {
    idx <- grep(section_pat, lines)
    if (length(idx) == 0) return(NA_real_)
    for (i in (idx[1]+1):min(idx[1]+8, length(lines))) {
      m <- regmatches(lines[i], regexpr("[0-9]+\\.[0-9]+", lines[i]))
      if (length(m) > 0) return(as.numeric(m))
    }
    NA_real_
  }

  cfi_line <- grep("^\\s+CFI\\s+[0-9]", lines, value = TRUE)[1]
  tli_line <- grep("^\\s+TLI\\s+[0-9]", lines, value = TRUE)[1]

  list(
    CFI   = as.numeric(regmatches(cfi_line, regexpr("[0-9]+\\.[0-9]+", cfi_line))),
    TLI   = as.numeric(regmatches(tli_line, regexpr("[0-9]+\\.[0-9]+", tli_line))),
    RMSEA = get_section_val("RMSEA \\(Root"),
    SRMR  = get_section_val("SRMR \\(Standard")
  )
}

out_files <- list.files("psychometrics/mimic", pattern = "model_mimic_.*\\.out$",
                        recursive = TRUE, full.names = TRUE)
out_files <- out_files[!grepl("archive", out_files)]
out_files <- out_files[!grepl("constraint", out_files)]
out_files <- out_files[sapply(out_files, function(f) {
  parts <- strsplit(normalizePath(f, winslash = "/"), "/")[[1]]
  idx <- max(which(parts == "demographics"))
  (length(parts) - idx) >= 2
})]

rows <- lapply(out_files, function(f) {
  fit <- tryCatch(parse_mplus_fit(f), error = function(e) list(CFI=NA, TLI=NA, RMSEA=NA, SRMR=NA))
  data.frame(
    Predictor = basename(dirname(f)),
    Model     = ifelse(grepl("unidim", basename(f), ignore.case = TRUE), "UNIDIM", "3-Factor"),
    CFI       = round(unlist(fit$CFI), 3),
    TLI       = round(unlist(fit$TLI), 3),
    RMSEA     = round(unlist(fit$RMSEA), 3),
    SRMR      = round(unlist(fit$SRMR), 3),
    stringsAsFactors = FALSE
  )
})

fit_result <- bind_rows(rows) |> arrange(Predictor, Model)

cat("\n========== MODEL FIT COMPARISON (3-Factor vs UNIDIM) ==========\n")
print(fit_result, row.names = FALSE, width = 100)

cat("\n========== STDY EFFECTS: Predictor -> Factor (3-Factor models) ==========\n")
cat("sig = p < .05  | ** = p < .01  | *** = p < .001\n\n")

out_3f <- out_files[!grepl("unidim", out_files, ignore.case = TRUE)]

eff_rows <- lapply(out_3f, function(f) {
  pred_folder <- basename(dirname(f))
  eff <- tryCatch(parse_stdy_effects(f), error = function(e) data.frame())
  if (nrow(eff) == 0) return(NULL)
  eff$ModelPredictor <- pred_folder
  eff
})

effects <- bind_rows(eff_rows) |>
  distinct() |>
  mutate(
    Sig = case_when(
      Pvalue < .001 ~ "***",
      Pvalue < .01  ~ "**",
      Pvalue < .05  ~ "*",
      TRUE          ~ ""
    )
  ) |>
  select(ModelPredictor, Factor, Predictor, Estimate, SE, EstSE, Pvalue, Sig) |>
  arrange(ModelPredictor, Factor, Predictor)

print(effects, row.names = FALSE, width = 120)

cat("\n========== STDY EFFECTS: Predictor -> COG (UNIDIM models) ==========\n")

out_uni <- out_files[grepl("unidim", out_files, ignore.case = TRUE)]

uni_rows <- lapply(out_uni, function(f) {
  pred_folder <- basename(dirname(f))
  eff <- tryCatch(parse_stdy_effects(f), error = function(e) data.frame())
  if (nrow(eff) == 0) return(NULL)
  eff$ModelPredictor <- pred_folder
  eff
})

uni_effects <- bind_rows(uni_rows) |>
  distinct() |>
  mutate(
    Sig = case_when(
      Pvalue < .001 ~ "***",
      Pvalue < .01  ~ "**",
      Pvalue < .05  ~ "*",
      TRUE          ~ ""
    )
  ) |>
  select(ModelPredictor, Factor, Predictor, Estimate, SE, EstSE, Pvalue, Sig) |>
  arrange(ModelPredictor, Predictor)

print(uni_effects, row.names = FALSE, width = 120)
