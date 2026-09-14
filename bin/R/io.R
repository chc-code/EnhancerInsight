suppressPackageStartupMessages({
  library(tidyverse); library(readr); library(DT); library(glue)
  library(knitr); library(kableExtra); library(htmltools)
  library(scales); library(ggrepel)
})
resolve_report_dir <- function(report_dir="report") {
  if (file.exists(file.path(report_dir,"ranking.tsv")) || file.exists(file.path(report_dir,"top20.tsv"))) return(normalizePath(report_dir,mustWork=FALSE))
  if (file.exists("ranking.tsv") || file.exists("top20.tsv")) return(normalizePath(".",mustWork=FALSE))
  if (file.exists(file.path(report_dir,"report","ranking.tsv")) || file.exists(file.path(report_dir,"report","top20.tsv"))) return(normalizePath(file.path(report_dir,"report"),mustWork=FALSE))
  normalizePath(report_dir,mustWork=FALSE)
}
read_report_tsv <- function(file, report_dir) {
  path <- file.path(report_dir,file)
  if (!file.exists(path)) return(NULL)
  readr::read_tsv(path,show_col_types=FALSE,progress=FALSE)
}
load_tables <- function(report_dir) list(
  summary=read_report_tsv("summary.tsv",report_dir),
  ranking=read_report_tsv("ranking.tsv",report_dir),
  top20=read_report_tsv("top20.tsv",report_dir),
  filtered_inputs=read_report_tsv("filtered_inputs.tsv",report_dir),
  rank_change=read_report_tsv("rank_change.tsv",report_dir),
  rank_change_largest=read_report_tsv("rank_change_largest.tsv",report_dir),
  rank_change_top20_union=read_report_tsv("rank_change_top20_union.tsv",report_dir),
  seed_contributions=read_report_tsv("seed_contributions.tsv",report_dir),
  direct_heat_flow=read_report_tsv("direct_heat_flow.tsv",report_dir)
)
detect_mode <- function(tables) {
  if (!is.null(tables$ranking) && "rwr_score_adjusted" %in% names(tables$ranking)) return("aware")
  if (!is.null(tables$rank_change) && all(c("input_score","input_rank","final_score","final_rank") %in% names(tables$rank_change))) return("independent_scores")
  "independent_equal"
}
final_score_col <- function(mode, ranking) {
  if (mode=="aware") return("rwr_score_adjusted")
  if (!is.null(ranking) && "rwr_score" %in% names(ranking)) return("rwr_score")
  if (!is.null(ranking) && "final_score" %in% names(ranking)) return("final_score")
  NULL
}
get_top_candidates <- function(tables) if (!is.null(tables$top20)) tables$top20 else tables$ranking
