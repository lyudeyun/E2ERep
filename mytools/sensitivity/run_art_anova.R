#!/usr/bin/env Rscript
# Helper: run ART-ANOVA on a CSV exported by hyperparam_sensitivity_anova.py
# Usage: Rscript run_art_anova.R <input.csv> <response_col> <output.csv> [ARTool_dir]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_art_anova.R input.csv response_col output.csv [ARTool_dir]")
}

input_csv <- args[1]
response_col <- args[2]
output_csv <- args[3]
artool_dir <- if (length(args) >= 4) args[4] else ""

if (!file.exists(input_csv)) {
  stop("Input CSV not found: ", input_csv)
}

if (nzchar(artool_dir) && dir.exists(artool_dir)) {
  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  if (!requireNamespace("ARTool", quietly = TRUE)) {
    devtools::install_local(artool_dir, upgrade = "never", quiet = TRUE)
  }
}

if (!requireNamespace("ARTool", quietly = TRUE)) {
  install.packages("ARTool", repos = "https://cloud.r-project.org", quiet = TRUE)
}

library(ARTool)

df <- read.csv(input_csv, stringsAsFactors = FALSE)
df$method <- factor(df$method)
df$fitness <- factor(df$fitness)
df$weight <- factor(df$weight)

if (!response_col %in% names(df)) {
  stop("Response column not found: ", response_col)
}

fml <- as.formula(paste(response_col, "~ method * fitness * weight"))
m <- art(fml, data = df)
a <- anova(m, type = "III")

out <- as.data.frame(a)
out$term <- rownames(a)
rownames(out) <- NULL

# Normalize column names toward Python output
names(out) <- gsub("Pr\\(>F\\)", "p_value", names(out), ignore.case = TRUE)
names(out) <- gsub("F.value", "F", names(out), ignore.case = TRUE)
names(out) <- gsub("Sum Sq", "sum_sq", names(out), ignore.case = TRUE)
names(out) <- gsub("Mean Sq", "mean_sq", names(out), ignore.case = TRUE)
names(out) <- gsub("Df", "df", names(out), ignore.case = TRUE)

write.csv(out, output_csv, row.names = FALSE)
cat("Wrote ART-ANOVA table to", output_csv, "\n")
