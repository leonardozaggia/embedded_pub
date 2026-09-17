# tetrachoric.R
# Computes the tetrachoric correlation matrix of the 23 Bayley-III items in the
# empirical three-factor CFA (WM, GDPS, FS) from the dHCP responses and saves it
# for Figure S5 (figures/supplementary/figS5_cosine_vs_tetrachoric.py), which
# compares embedding cosine similarities with the empirical item co-variation.
#
# Input : psychometrics/combined.dat + psychometrics/varnames.txt
#         (built by psychometrics/prepare_data.R from the restricted dHCP data)
# Output: data/outputs/tetrachoric/tetrachoric_matrix.csv
#         data/outputs/tetrachoric/cfa_item_factor_map.csv
#
# Run from anywhere:  Rscript analysis/tetrachoric.R
# Requires the R packages psych, dplyr, readr.

suppressPackageStartupMessages({
  library(psych)
  library(dplyr)
  library(readr)
})

# ── locate the repository root from this script's own path ──────────────────
script_path <- {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) normalizePath(sub("^--file=", "", m[1])) else normalizePath("analysis/tetrachoric.R")
}
ROOT <- dirname(dirname(script_path))
OUT  <- file.path(ROOT, "data/outputs/tetrachoric")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

MISSING <- -999

# ── load combined.dat + varnames ──────────────────────────────────────────────
varnames <- readLines(file.path(ROOT, "psychometrics/varnames.txt"))
varnames <- trimws(varnames[nchar(trimws(varnames)) > 0])

dat <- read.table(file.path(ROOT, "psychometrics/combined.dat"),
                  header = FALSE, col.names = varnames, na.strings = as.character(MISSING))

# ── the 23-item CFA assignments, parsed from psychometrics/measurement/model_cfa_3factor.inp
inp <- readLines(file.path(ROOT, "psychometrics/measurement/model_cfa_3factor.inp"))
model_block <- inp[seq(grep("^MODEL:", inp), grep("^OUTPUT:", inp) - 1)]
model_block <- model_block[!grepl("^\\s*!", model_block)]          # drop comments
model_text  <- paste(model_block, collapse = " ")
by_stmts    <- regmatches(model_text, gregexpr("(WM|GDP|FS)\\s+BY[^;]*;", model_text))[[1]]
factor_name <- c(WM = "working_memory", GDP = "goal_directed_problem_solving", FS = "flexibility_shift")
fa <- bind_rows(lapply(by_stmts, function(stmt) {
  f     <- sub("\\s+BY.*", "", trimws(stmt))
  items <- regmatches(stmt, gregexpr("COG[0-9]{3}", stmt))[[1]]
  data.frame(item_col = items, assigned_factor = factor_name[[f]], stringsAsFactors = FALSE)
}))
stopifnot(nrow(fa) == 23)

ordered_items <- fa$item_col
item_df <- dat[, ordered_items]
cat("CFA items per factor:\n")
print(table(fa$assigned_factor))

# ── tetrachoric correlation ───────────────────────────────────────────────────
X <- item_df[, ordered_items]
tet <- psych::tetrachoric(X)
R_tet <- tet$rho

# ── save ─────────────────────────────────────────────────────────────────────
write.csv(R_tet, file.path(OUT, "tetrachoric_matrix.csv"), row.names = TRUE)
write.csv(fa[, c("item_col", "assigned_factor")],
          file.path(OUT, "cfa_item_factor_map.csv"), row.names = FALSE)

cat("Saved tetrachoric matrix and factor map to:\n", OUT, "\n")
