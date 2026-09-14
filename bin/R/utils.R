fmt_num <- function(x,digits=5) ifelse(is.na(x),NA,signif(as.numeric(x),digits))
fmt_df <- function(df,digits=5) df %>% mutate(across(where(is.numeric),~fmt_num(.x,digits)))
drop_all_na_cols <- function(df) df[,vapply(df,function(x)!all(is.na(x)|x==""),logical(1)),drop=FALSE]
pretty_names <- function(df) {
  nm <- names(df)
  nm <- dplyr::recode(nm,
    rank="Rank",input_region="Region",input_score="Input score",
    rwr_score="Final RWR score",rwr_score_adjusted="Final RWR score",final_score="Final RWR score",
    celltype="Cell type",n_contributing_seeds="No. of seeds with contribution >1%",n_seeds_reached="No. of seeds with contribution >1%",
    direct_interacting_genes="Direct interacting genes",
    direct_interacting_genes_short="Direct interacting genes",covered_node_genes="Covered genes",
    display_gene="Gene",
    input_rank="Input rank",final_rank="Final rank",rank_change="Rank change",
    seed_region="Seed region",seed_node="Seed node",seed_label="Seed",seed_heat="Initial heat",
    seed_input_score="Input score",contribution="Contribution",
    contribution_fraction="Contribution (%)",graph_distance="Distance",
    focal_node="Candidate-driving node",focal_node_label="Candidate-driving node",
    neighbor_node="Neighbor node",neighbor_label="Direct heat recipient",
    edge_weight="Edge weight",transition_probability="Transition probability",
    estimated_heat_outflow="Estimated heat outflow",gwas_snps="GWAS SNPs",gwas_disease_traits="GWAS disease traits",
    in_input_top20="Input Top20",in_final_top20="Final Top20",.default=nm)
  names(df) <- nm
  df
}
dt_tbl <- function(df,caption=NULL,page_length=20,max_rows=200) {
  if (is.null(df)||nrow(df)==0) {cat("No data available."); return(invisible(NULL))}
  if (nrow(df)>max_rows) df <- df %>% slice_head(n=max_rows)
  DT::datatable(df,caption=caption,rownames=FALSE,filter="top",
    options=list(pageLength=page_length,scrollX=TRUE,autoWidth=TRUE,deferRender=TRUE))
}
download_links <- function(files,report_dir) {
  files <- files[file.exists(file.path(report_dir,files))]
  if (length(files)==0) return(invisible(NULL))
  htmltools::tagList(htmltools::tags$p(htmltools::tags$strong("Download full file(s): "),
    lapply(seq_along(files),function(i) htmltools::tagList(
      if(i>1) htmltools::HTML(" &nbsp;|&nbsp; "),htmltools::tags$a(href=files[[i]],files[[i]])))))
}
method_summary <- function(mode) {
  if (mode=="independent_equal") return("Context-independent prioritization: Random Walk with Restart (RWR) on the pre-built network, with all input regions as equal-weight seeds.")
  if (mode=="independent_scores") return("Context-independent prioritization: heat-weighted RWR on the pre-built network, using user-provided input scores as seed heat.")
  if (mode=="aware") return("Context-aware prioritization: RWR on a cell type-specific rewired network. Dynamic edge weight equals base edge weight multiplied by a specificity modifier.")
  "Prioritization workflow."
}
mode_label <- function(mode) dplyr::case_when(mode=="independent_equal"~"Context-independent (equal input weights)",mode=="independent_scores"~"Context-independent (input scores)",mode=="aware"~"Context-aware",TRUE~mode)


compact_semicolon_text <- function(x, max_items = 8, empty = "") {
  x <- as.character(x)

  vapply(x, function(value) {
    if (is.na(value) || trimws(value) == "") return(empty)

    items <- trimws(unlist(strsplit(value, ";", fixed = TRUE)))
    items <- unique(items[items != ""])

    if (length(items) == 0) return(empty)

    result <- paste(head(items, max_items), collapse = "; ")

    if (length(items) > max_items) {
      result <- paste0(result, " (+", length(items) - max_items, " more)")
    }

    result
  }, character(1))
}


compact_semicolon_html <- function(x, max_items = 8, empty = "") {
  x <- as.character(x)

  vapply(x, function(value) {
    if (is.na(value) || trimws(value) == "") {
      return(empty)
    }

    items <- trimws(unlist(strsplit(value, ";", fixed = TRUE)))
    items <- unique(items[items != ""])

    if (length(items) == 0) {
      return(empty)
    }

    shown <- head(items, max_items)
    shown <- htmltools::htmlEscape(shown)

    main_text <- paste(shown, collapse = "; ")

    if (length(items) > max_items) {
      remaining <- length(items) - max_items
      paste0(
        main_text,
        "<br><span class=\"gwas-more\">(+",
        remaining,
        " more)</span>"
      )
    } else {
      main_text
    }
  }, character(1))
}
