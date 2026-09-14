# EnhancerInsight

**EnhancerInsight** integrates multiresolution and multiomics evidence to characterize and prioritize candidate enhancers within a context-aware framework.
The functional characterization module is to define the regulatory context and biological relevance of candidate enhancers, leveraging genomic location and support from existing enhancer databases to establish enhancer identity, bulk and single-cell enhancer activity to infer tissue and cell-type origins, and TF binding, enhancer–gene regulation, disease-associated variants, eQTLs, and downstream effects to elucidate their regulatory and functional roles.
The prioritization module uses network propagation over context-independent or context-aware regulatory networks to rank candidate regulatory regions. Context-independent networks prioritize highly connected enhancers regardless of cellular context, whereas context-aware networks rewire regulatory connections based on cell-type specificity, thereby identifying enhancers with both strong regulatory connectivity and cell-type-specific relevance.

EnhancerInsight is available both as a web server for easy access and a standalone toolkit for large-scale or customized analyses. The web server is available at **https://bioinfo.vanderbilt.edu/enhancerinsight/**.

Currently, only **hg38** is supported.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Reference databases](#reference-databases)
- [Usage](#usage)
  - [1. Characterization](#1-characterization)
  - [2. Generating the characterization report](#2-generating-the-characterization-report)
  - [3. Prioritization — context-independent](#3-prioritization--context-independent)
  - [4. Prioritization — context-aware](#4-prioritization--context-aware)
  - [5. Generating the prioritization report](#5-generating-the-prioritization-report)
- [Output files](#output-files)
- [Input format](#input-format)
- [Citation](#citation)
- [License](#license)

---

## Requirements

- Linux or macOS (tested on Ubuntu 20.04+)
- [Pixi](https://pixi.sh) (recommended) **or** Python ≥ 3.9 + R ≥ 4.2 installed manually
- bedtools ≥ 2.29 (included in the pixi environment)
- ~12 GB free disk space for reference databases

---

## Installation

### Recommended: pixi (reproducible environment)

[Pixi](https://pixi.sh) manages both Python and R dependencies in a fully
reproducible conda-based environment defined by `pixi.toml`.

**1. Install pixi**

```bash

# Optional, set the PIXI_HOME to other folders (default is install to HOME), e.g. /data/pixi
export PIXI_HOME="/data/pixi"  # if set, please make sure you change the actual path and add to your shell  rc file
echo "export PIXI_HOME='/data/pixi'" >> ~/.bashrc  # if you are using bash
echo "export PIXI_HOME='/data/pixi'" >> ~/.zshrc  # if you are using zsh


curl -fsSL https://pixi.sh/install.sh | sh
# Re-open your shell or run:
source ~/.bashrc   # or ~/.zshrc
```

**2. Clone the repository**

```bash
git clone https://github.com/chc-code/EnhancerInsight.git
cd EnhancerInsight
```



**3. Create and install the environment**
```bash
# if you are using MacOS, need to replace the pixi.toml with pixi.macos.toml
cp pixi.macos.toml pixi.toml

# install all the packages
pixi install


```

This reads `pixi.toml` and installs all Python packages, R, and system tools
(including bedtools) into an isolated environment under `.pixi/envs/default/`.

**4. Verify the installation**

```bash
pixi run python bin/annoEnhancer.py --help
pixi run python bin/random_walk_rank_regions_context_independent.py --help
pixi run python bin/random_walk_rank_regions_context_aware.py --help
pixi run Rscript -e "library(rmdformats); cat('R env OK\n')"
```

### Alternative: manual installation

If you prefer to manage dependencies yourself:

```bash
# Python packages
pip install pandas numpy networkx scipy pybedtools

# R packages (run inside R)
install.packages(c("rmarkdown", "rmdformats", "DT", "dplyr",
                   "ggplot2", "VennDiagram", "pheatmap",
                   "RColorBrewer", "gplots", "data.table"))

# Optional (for better label placement / colour palettes in the report)
install.packages(c("ggrepel", "WebGestaltR"))
```

bedtools must be available on your `PATH`:

```bash
# Ubuntu/Debian
sudo apt install bedtools

# macOS (Homebrew)
brew install bedtools
```

---

## Reference databases

Reference databases are required for characterization and prioritization and must be
downloaded separately using the links below.

| Name | Download |
|---|---|
| Reference files (hg38) | [download](https://www.6157777.xyz/enhancerinsight/download/enhancer_insight.hg38.tgz) |
| Pre-built network files | [download](https://www.6157777.xyz/enhancerinsight/download/enhancer_insight.network.tgz) |


After downloading, extract to a directory (referred to as `<refdir>` below):

```bash
tar -xzf enhancer_insight.hg38.tgz -C /path/to/refdir/
tar -xzf enhancer_insight.network.tgz -C /path/to/refdir/
```

Pre-built network files for prioritization:

| File | Used by | Description |
|---|---|---|
| `prebuilt_network_v2_with_gwas.gpickle` | both | Unified regulatory network |
| `dbscATAC_edgeprep_compact.base_edges.npz` | context-aware | Base edge matrix |
| `dbscATAC_edgeprep_compact.edge_override/<CellType>.npz` | context-aware | Per-cell-type edge weight overrides |
| `node_specificity_dbscATAC.nodes.tsv` | context-aware | Node ID ↔ index mapping |
| `base_graph_topology.node_topology.tsv.gz` | context-aware | Node topology (component size, coreness) |

Place these files in the same `<refdir>` or a dedicated `bin/` directory and
update the paths in the commands below accordingly.

---

## Usage

For pixi users, prefix every command with `pixi run`. If you are using a manual
installation, run the scripts with your system Python/Rscript directly.

### 1. Characterization

```bash
pixi run python bin/annoEnhancer.py \
    -m hg38 \
    -in input.bed \
    -o MyProject \
    -w ./results/
```

**All options:**

| Flag | Default | Required | Description |
|---|---|---|---|
| `-in` | — | **yes** | Input BED file (≥ 3 columns: chr, start, end), hg38 coordinates |
| `-m` | `hg38` | no | Genome assembly. Currently only `hg38` is supported |
| `-o` | `annoEnhancer` | no | Output prefix (project name) |
| `-w` | — | **yes** | Output/working directory |
| `-cb` | — | no | Custom binding-sites BED (adds a `user_binding` Y/N column) |
| `-pri` | `0` | no | Set to `1` to add a database-support priority score column |
| `-t` | — | no | Number of threads (activated when input > 100,000 lines) |
| `-maxt` | `8` | no | Maximum allowed threads |
| `-minl` | — | no | Minimum number of input regions/lines |

**Example with all options:**

```bash
pixi run python bin/annoEnhancer.py \
    -m hg38 \
    -in regions.bed \
    -o GM12878_H3K27ac \
    -w ./results/ \
    -cb chipseq_peaks.bed \
    -pri 1 \
    -t 8
```

---

### 2. Generating the characterization report

After running characterization, generate the interactive HTML report using the
provided R Markdown template:

```bash
pixi run python bin/render_annoEnhancer_report_updated.py \
    --rmd    bin/report-update.Rmd \
    --outdir ./results/ \
    --proj   MyProject \
    --genome hg38 \
    --out    ./results/MyProject_report.html
```

Alternatively, render directly from R:

```r
rmarkdown::render(
  "bin/report-update.Rmd",
  params = list(
    outdir  = "./results/",
    proj    = "MyProject",
    genome  = "hg38",
    refdir  = "/path/to/refdir/hg38/"
  ),
  output_file = "./results/MyProject_report.html"
)
```

The generated report is self-contained and can be shared without any server
dependency. It includes:

1. **Summary** — genomic feature distribution; support by public super-enhancer
   and enhancer databases; Venn diagrams; VISTA validation overlap
2. **Activity specificity** — Manhattan-style plot of cell-line activity
   (HACER GRO-seq signal, MeanZ across input regions)
3. **Cell-type specificity** — scATAC-seq-based Manhattan plot
   (NormalizedSumZ corrected for enhancer count per cell type)
4. **TF binding enrichment** — heatmaps (Jaccard, co-occurrence, input frequency)
5. **Functional enrichment** — KEGG pathway enrichment of predicted target genes
   (five assignment methods: closest, 50 kb, FANTOM5, validated, eQTL)
6. **GWAS SNP enrichment** — trait-level Jaccard index table
7. **GTEx eQTL enrichment** — tissue-level Jaccard index table

---

### 3. Prioritization — context-independent

For each candidate, the user-provided importance score (e.g., GWAS −log₁₀(*p*-value)) is assigned as its initial seed score. When no importance scores are provided, all candidate nodes are assigned equal initial scores. Seed scores are then propagated through the unified regulatory network using the source-integrated interaction weights, allowing scores to diffuse across connected genes and regulatory elements.

```bash
pixi run python bin/random_walk_rank_regions_context_independent.py \
    --network /path/to/prebuilt_network_v2_with_gwas.gpickle \
    --input-regions input.bed \
    --restart 0.5 \
    --tol 1e-7 \
    --max-iter 100 \
    --combine-overlaps mean \
    --output output_context_independent \
    --report-top-n 20 \
    --report-top-network-n 3 \
    --report-max-neighbors 25
```

**All options:**

| Flag | Default | Required | Description |
|---|---|---|---|
| `--network` | — | **yes** | Pre-built network file (.gpickle) |
| `--input-regions` | — | **yes** | Input regions, see [Input format](#input-format) |
| `--output` | — | **yes** | Output directory (`ranking.tsv` and `report/` are written here) |
| `--restart` | `0.5` | no | RWR restart probability (0–1). Higher = stronger return to the initial scores |
| `--tol` | `1e-7` | no | L1 convergence tolerance |
| `--max-iter` | `100` | no | Maximum RWR iterations |
| `--combine-overlaps` | `max` | no | How to combine scores when a region overlaps multiple nodes: `max`, `mean`, `union` |
| `--local-support-eta` | `0.3` | no | Weight of local seed support in the final score |
| `--local-support-max-hops` | `3` | no | Maximum hops used for local seed support |
| `--local-support-hop-decay` | `0.5` | no | Multiplicative decay for each additional local-support hop |
| `--annotate-top-n` | `0` | no | If > 0, add component-aware annotations for the top N ranked regions |
| `--tiny-component-size` | `2` | no | Component size threshold for tiny components |
| `--report-top-n` | `20` | no | Number of top regions exported for the report |
| `--report-top-network-n` | `3` | no | Number of top regions for local network export |
| `--report-max-neighbors` | `25` | no | Maximum neighbors per overlapped node in local network export |

---

### 4. Prioritization — context-aware

For each candidate, the user-provided importance score (e.g., GWAS −log₁₀(*p*-value)) is assigned as its initial seed score. When no importance scores are provided, all candidate nodes are assigned equal initial scores. Seed scores are then propagated through the corresponding cell-type-specific rewired network, in which interaction weights are adjusted according to the specificity scores of the connecting nodes.

Available cell types correspond to the `.npz` files in the `edge_override/` directory.

```bash
pixi run python bin/random_walk_rank_regions_context_aware.py \
    --network       /path/to/prebuilt_network_v2_with_gwas.gpickle \
    --base-edges    /path/to/dbscATAC_edgeprep_compact.base_edges.npz \
    --edge-override /path/to/dbscATAC_edgeprep_compact.edge_override/Cardiomyocyte.npz \
    --nodes-index   /path/to/node_specificity_dbscATAC.nodes.tsv \
    --topology-file /path/to/base_graph_topology.node_topology.tsv.gz \
    --input-regions input.bed \
    --restart 0.5 \
    --combine-overlaps max \
    --output output_context_aware_Cardiomyocyte \
    --report-top-n 20 \
    --report-top-network-n 3 \
    --report-max-neighbors 25
```

**All options:**

| Flag | Default | Required | Description |
|---|---|---|---|
| `--base-edges` | — | **yes** | Base regulatory network edge matrix (.npz) |
| `--edge-override` | — | **yes** | Cell-type-specific edge weight matrix (.npz) |
| `--nodes-index` | — | **yes** | Node ID ↔ index mapping (.tsv) |
| `--topology-file` | — | **yes** | Node topology file (`node_id`, `component_size`[, `coreness`]) |
| `--input-regions` | — | **yes** | Input regions, see [Input format](#input-format) |
| `--output` | — | **yes** | Output directory (`ranking.tsv` and `report/` are written here) |
| `--network` | — | no | Pre-built network file (.gpickle), as in the example above |
| `--min-component-size` | `5` | no | Nodes in network components smaller than this are filtered out |
| `--use-coreness-penalty` | `off` | no | Node-level coreness penalty after component filtering: `off`, `linear`, `sqrt` |
| `--restart` | `0.5` | no | RWR restart probability (0–1) |
| `--tol` | `1e-9` | no | Convergence tolerance |
| `--max-iter` | `100` | no | Maximum RWR iterations |
| `--combine-overlaps` | `max` | no | Score combination strategy: `max`, `mean`, `union` |
| `--report-top-n` | `20` | no | Number of top regions exported for the report |
| `--report-top-network-n` | `3` | no | Number of top regions for local network export |
| `--report-max-neighbors` | `25` | no | Maximum neighbors per overlapped node in local network export |
| `--verbose` | off | no | Print progress details |

**List available cell types:**

```bash
ls /path/to/dbscATAC_edgeprep_compact.edge_override/*.npz | xargs -n1 basename | sed 's/.npz//'
```

---

### 5. Generating the prioritization report

Both prioritization scripts write report-ready files to `<output>/report/`.
Render them into an HTML report with:

```bash
cd bin
pixi run Rscript render_report.R /path/to/output_context_aware_Cardiomyocyte/report/ prioritization_report.html
```

The first argument is the report directory, the second the HTML file name.
The HTML report is written into the report directory.

---

## Output files

### Characterization

| File | Description |
|---|---|
| `<prefix>.annotation.txt` | Main tab-separated annotation table (one row per input region) |
| `<prefix>-HACER.txt` | GRO-seq activity per dataset per region (used in activity specificity plot) |
| `<prefix>-dbscATAC.txt` | scATAC-seq accessibility per cell type per region |
| `<prefix>.m_tf.txt` | TF binding enrichment counts (genomic feature × cell line × TF) |
| `<prefix>.m_SNP.txt` | GWAS variant overlap counts per trait |
| `<prefix>.m_eQTL.txt` | GTEx eQTL overlap counts per tissue |
| `<prefix>.target.txt` | Target gene lists (5 rows: closest / 50 kb / FANTOM5 / validated / eQTL) |
| `<prefix>_report.html` | Self-contained interactive HTML report |

### Characterization output columns

| Column | Description |
|---|---|
| `chr / start / end` | Input coordinates (0-based, half-open) |
| `Function-NCBI_RefSeq` | Genomic feature category |
| `NearestGene` | Nearest gene(s) by TSS distance |
| `Genes-within-50kb` | All genes with TSS within 50 kb |
| `superEnhancer-SEdb2.0` | Y/N — SEdb 2.0 overlap |
| `superEnhancer-dbSUPER` | Cell line identifier(s) from dbSUPER |
| `Enhancer-HACER` | Cell line identifier(s) from HACER |
| `Enhancer-VISTA` | Tissue(s) from VISTA |
| `Enhancer-RAEdb` | Tissue(s) from RAEdb |
| `Enhancer-EnsemblRegulatoryBuild` | Y/NA |
| `Enhancer-ENCODEBroad-ChromHMM` | Y/NA |
| `Enhancer-NIH-ROADMAP` | Tissue(s) from NIH Roadmap |
| `targetGene-experimental` | Validated target genes (source:gene format) |
| `targetGene-FANTOM5` | FANTOM5 target genes |
| `TF binding sites` | Overlapping TF binding sites (cellline:TF format) |
| `GWAS SNPs` | Overlapping GWAS variants (trait:rsID format) |
| `eQTL variants` | Overlapping GTEx eQTL variants |
| `user_binding` | Y/N — user-supplied binding site overlap (only if `-cb` used) |

### Prioritization

| Path | Description |
|---|---|
| `<output>/ranking.tsv` | Ranked table, one row per input region, with the input score (if provided) and RWR scores |
| `<output>/report/` | Report-ready files (top `--report-top-n` regions, local networks of the top `--report-top-network-n` regions) |
| `<output>/report/<name>.html` | HTML report generated by `render_report.R` |

---

## Input format

### Characterization

Standard **BED format** (tab-separated), hg38 coordinates. Columns 1–3 are required;
coordinates follow the 0-based, half-open convention used by UCSC and bedtools.

```
chr1    713441    714434
chr8    127742018 127744200
```

> A region spanning bases 1000–2000 (1-based, inclusive) is written as
> `999  2000` in BED (0-based, exclusive end).

### Prioritization

Regions or variants in hg38, whitespace-separated. The importance score is optional;
when it is missing, all candidates start with equal scores. Extra columns are ignored,
and lines starting with `#` are skipped.

| Layout | Columns | Score column |
|---|---|---|
| Interval (BED) | `chr  start  end  [score]` | 4 |
| Variant (SNP) | `chr  pos  [score]` | 3 |
| Interval string | `chr:start-end  [score]` | 2 |
| Variant string | `chr:pos  [score]` | 2 |

The whole file is read as either interval or SNP layout: if any row has no valid
integer end coordinate in column 3 (missing, non-integer, or smaller than column 2),
the file is treated as SNP layout.

```
# Interval layout, score in column 4
chr1    713441    714434    24.15
chr8    127742018 127744200 13.70

# SNP layout, score in column 3, extra gene column ignored
chr2    71392981    9.52    ZNF638
chr2    174648092   41.22   WIPF1
```

---

## Citation

To be published...

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
