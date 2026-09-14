
safe_chr <- function(x, fallback = "") {
  if (length(x) == 0 || is.null(x) || all(is.na(x))) return(fallback)

  value <- as.character(x[[1]])

  if (is.na(value) || value == "") fallback else value
}

scale_node_size <- function(x, to = c(34, 58), fallback = 42) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x) | x < 0] <- 0

  if (length(x) == 0) return(numeric())
  if (all(x == 0)) return(rep(fallback, length(x)))

  transformed <- sqrt(x)
  rg <- range(transformed, na.rm = TRUE)

  if (!all(is.finite(rg)) || diff(rg) == 0) {
    return(rep(mean(to), length(x)))
  }

  scales::rescale(transformed, to = to)
}

scale_edge_width <- function(x, to = c(1.5, 9)) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x) | x < 0] <- 0

  if (length(x) == 0) return(numeric())
  if (all(x == 0)) return(rep(mean(to), length(x)))

  transformed <- sqrt(x)
  rg <- range(transformed, na.rm = TRUE)

  if (!all(is.finite(rg)) || diff(rg) == 0) {
    return(rep(mean(to), length(x)))
  }

  scales::rescale(transformed, to = to)
}

cytoscape_network_html <- function(region, nodes, edges, seeds = NULL, widget_id) {
  stopifnot(is.character(widget_id), length(widget_id) == 1)
  nodes_plot <- as.data.frame(nodes)
  edges_plot <- as.data.frame(edges)

  # Add the existing seed annotations to seed nodes for hover/click details.
  if (!is.null(seeds) && nrow(seeds) > 0 && "seed_node" %in% names(seeds)) {
    seed_extra <- seeds %>%
      mutate(node_id = as.character(seed_node)) %>%
      select(any_of(c("node_id", "contribution", "contribution_fraction", "graph_distance",
                      "gwas_snps", "gwas_disease_traits"))) %>%
      distinct(node_id, .keep_all = TRUE)
    nodes_plot <- nodes_plot %>% left_join(seed_extra, by = "node_id")
  }

  focal_id <- ""
  if (!is.null(seeds) && nrow(seeds) > 0 && "focal_node" %in% names(seeds)) {
    focal_id <- safe_chr(seeds$focal_node, fallback = "")
  }
  if (focal_id == "" && "role" %in% names(nodes_plot)) {
    # The focal node is a seed when present; otherwise keep the first node as fallback.
    focal_id <- safe_chr(nodes_plot$node_id, fallback = region)
  }

  node_elements <- lapply(seq_len(nrow(nodes_plot)), function(i) {
    r <- nodes_plot[i, , drop = FALSE]
    is_focal <- identical(as.character(r$node_id[[1]]), focal_id)
    dat <- list(
      id = as.character(r$node_id[[1]]),
      label = safe_chr(r$label, fallback = as.character(r$node_id[[1]])),
      role = safe_chr(r$role, fallback = "intermediate"),
      node_size = suppressWarnings(as.numeric(r$node_size[[1]])),
      node_color = safe_chr(r$node_color, fallback = "#D9D9D9"),
      node_shape = safe_chr(r$node_shape, fallback = "ellipse"),
      `Display name` = safe_chr(r$label, fallback = as.character(r$node_id[[1]])),
      Region = as.character(r$node_id[[1]]),
      `Node ID` = as.character(r$node_id[[1]]),
      `Gene ID` = if ("gene_id" %in% names(r)) safe_chr(r$gene_id, "") else "",
      `Gene name` = if ("gene_name" %in% names(r)) safe_chr(r$gene_name, "") else "",
      `Initial heat` = if ("initial_heat" %in% names(r)) suppressWarnings(as.numeric(r$initial_heat[[1]])) else NA_real_,
      Distance = if ("graph_distance" %in% names(r)) safe_chr(r$graph_distance, "") else "",
      `GWAS SNPs` = if ("gwas_snps" %in% names(r)) safe_chr(r$gwas_snps, "") else "",
      `GWAS disease traits` = if ("gwas_disease_traits" %in% names(r)) safe_chr(r$gwas_disease_traits, "") else ""
    )
    list(group="nodes", data=dat, classes=paste(c(dat$role, if(is_focal) "focal" else NULL), collapse=" "))
  })

  edge_elements <- lapply(seq_len(nrow(edges_plot)), function(i) {
    r <- edges_plot[i, , drop=FALSE]
    w <- if ("dynamic_weight" %in% names(r)) suppressWarnings(as.numeric(r$dynamic_weight[[1]])) else if ("weight" %in% names(r)) suppressWarnings(as.numeric(r$weight[[1]])) else 1
    list(group="edges", data=list(id=paste0(widget_id,"_edge_",i), source=as.character(r$source[[1]]), target=as.character(r$target[[1]]),
                                   width=max(1.5, min(8, 1.5 + 6.5*sqrt(max(w,0)))),
                                   `Edge weight`=w,
                                   `Edge type`=if("edge_type" %in% names(r)) safe_chr(r$edge_type,"") else ""))
  })

  elements_json <- jsonlite::toJSON(c(node_elements, edge_elements), auto_unbox=TRUE, dataframe="rows", na="null", null="null", digits=12)
  graph_id <- paste0(widget_id,"_graph"); side_id <- paste0(widget_id,"_details"); fit_id <- paste0(widget_id,"_fit"); reset_id <- paste0(widget_id,"_reset"); export_id <- paste0(widget_id,"_export")
  graph_id_json <- jsonlite::toJSON(graph_id,auto_unbox=TRUE); side_id_json <- jsonlite::toJSON(side_id,auto_unbox=TRUE)
  fit_id_json <- jsonlite::toJSON(fit_id,auto_unbox=TRUE); reset_id_json <- jsonlite::toJSON(reset_id,auto_unbox=TRUE); export_id_json <- jsonlite::toJSON(export_id,auto_unbox=TRUE)
  png_name_json <- jsonlite::toJSON(paste0(gsub("[^A-Za-z0-9_-]","_",region),".png"),auto_unbox=TRUE)

  html <- glue::glue('
<div class="cy-report-toolbar"><button id="{fit_id}" type="button">Fit network</button><button id="{reset_id}" type="button">Reset view</button><button id="{export_id}" type="button">Export PNG</button></div>
<div class="cy-report-legend">
  <div><span class="cy-dot" style="background:#3182BD"></span>Seed / focal (blue intensity = initial heat) &nbsp;&nbsp; <span class="cy-dot" style="background:#D9D9D9"></span>Intermediate node</div>
  <div style="margin-top:6px;">○ Gene node &nbsp;&nbsp; △ Enhancer / regulatory-region node &nbsp;&nbsp; <span class="cy-line" style="border-top:2px solid #7F7F7F"></span>Selected focal-to-seed path</div>
</div>
<div class="cy-report-wrapper"><div id="{graph_id}" class="cy-report-network"></div><div id="{side_id}" class="cy-report-side"><strong>Node or edge details</strong><p>Click a node or edge in the network.</p></div></div>
<script>
(function() {{
 var container=document.getElementById({graph_id_json}), details=document.getElementById({side_id_json});
 var fitButton=document.getElementById({fit_id_json}), resetButton=document.getElementById({reset_id_json}), exportButton=document.getElementById({export_id_json});
 function esc(v){{if(v===null||v===undefined||v==="")return "";return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}}
 function formatValue(v){{if(v===null||v===undefined||v==="")return "";if(typeof v==="number")return Number(v).toPrecision(6);return esc(v).replace(/;/g,";<wbr> ");}}
 window.{widget_id}=cytoscape({{container:container,elements:{elements_json},layout:{{name:"cose",fit:true,padding:45,animate:false}},minZoom:0.15,maxZoom:4,wheelSensitivity:0.18,
 style:[
  {{selector:"node",style:{{"label":"data(label)","font-size":11,"text-wrap":"wrap","text-max-width":120,"text-valign":"bottom","text-margin-y":7,"background-color":"data(node_color)","border-color":"#333333","border-width":1,"width":"data(node_size)","height":"data(node_size)","shape":"data(node_shape)"}}}},
  {{selector:"node.seed",style:{{"font-weight":"bold"}}}},
  {{selector:"edge",style:{{"curve-style":"bezier","width":"data(width)","opacity":0.75,"line-color":"#7F7F7F","target-arrow-shape":"none"}}}},
  {{selector:":selected",style:{{"border-color":"#ff9800","border-width":4,"line-color":"#ff9800"}}}}
 ]}});
 if(fitButton)fitButton.addEventListener("click",function(){{window.{widget_id}.fit(undefined,40);}});
 if(resetButton)resetButton.addEventListener("click",function(){{window.{widget_id}.reset();window.{widget_id}.fit(undefined,40);}});
 if(exportButton)exportButton.addEventListener("click",function(){{var link=document.createElement("a");link.href=window.{widget_id}.png({{full:true,scale:2,bg:"#ffffff"}});link.download={png_name_json};document.body.appendChild(link);link.click();document.body.removeChild(link);}});
 function showElementDetails(target){{var data=target.data();var title=target.isNode()?(data.label||data.id||"Node"):"Network edge";var hidden={{id:true,source:true,target:true,width:true,node_size:true,node_color:true,node_shape:true,label:true,role:true}};var rows=[];Object.keys(data).forEach(function(key){{if(hidden[key])return;var value=formatValue(data[key]);if(value!=="")rows.push("<tr><th style=\'text-align:left;vertical-align:top;padding-right:8px;\'>"+esc(key)+"</th><td>"+value+"</td></tr>");}});details.innerHTML="<strong>"+esc(title)+"</strong><table style=\'margin-top:8px;width:100%;font-size:0.9em;\'>"+rows.join("")+"</table>";}}
 window.{widget_id}.on("tap","node, edge",function(evt){{showElementDetails(evt.target);}});window.{widget_id}.on("mouseover","node",function(evt){{showElementDetails(evt.target);}});window.{widget_id}.on("tap",function(evt){{if(evt.target===window.{widget_id})details.innerHTML="<strong>Node or edge details</strong><p>Click a node or edge in the network.</p>";}});
}})();
</script>', .open="{", .close="}")
  htmltools::HTML(html)
}
