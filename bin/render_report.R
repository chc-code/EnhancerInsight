args <- commandArgs(trailingOnly = TRUE)
report_dir <- ifelse(length(args) >= 1, args[[1]], "report")
output_file <- ifelse(length(args) >= 2, args[[2]], "prioritization_report.html")
resolve_output_dir <- function(report_dir) {
  if (file.exists(file.path(report_dir, "ranking.tsv")) || file.exists(file.path(report_dir, "top20.tsv"))) return(report_dir)
  if (file.exists("ranking.tsv") || file.exists("top20.tsv")) return(".")
  if (file.exists(file.path(report_dir, "report", "ranking.tsv")) || file.exists(file.path(report_dir, "report", "top20.tsv"))) return(file.path(report_dir, "report"))
  report_dir
}
rmarkdown::render(
  input = "master_prioritization_report.Rmd",
  output_dir = resolve_output_dir(report_dir),
  output_file = output_file,
  params = list(report_dir = report_dir),
  envir = new.env(parent = globalenv())
)
