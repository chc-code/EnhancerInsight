# EnhancerInsight

**EnhancerInsight** is a platform for high-resolution, context-aware annotation and prioritization of candidate enhancers.
By integrating single-cell and multi-omics data, it reveals cell-type–specific regulatory activity and reconstructs enhancer-centered regulatory context, including TF programs, enhancer–gene links, and variant associations. A network-based prioritization framework enables ranking of candidate regions using traits, cell types, or custom scores, supporting biologically informed discovery in disease-relevant contexts.

EnhancerInsight is available both as a web server for easy access and a standalone toolkit for large-scale or customized analyses. The web server is available at **https://www.6157777.xyz/enhancerinsight/**.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Reference databases](#reference-databases)
- [Usage](#usage)
  - [Annotation](#1-annotation)
  - [Prioritization — Mode 1: trait-guided](#2-prioritization--mode-1-trait-guided)
  - [Prioritization — Mode 2A: input signal](#3-prioritization--mode-2a-input-signal)
  - [Prioritization — Mode 2B: cell-type specific](#4-prioritization--mode-2b-cell-type-specific)
  - [Generating the HTML report](#5-generating-the-html-report)
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
pixi run python bin/annoEnhancer_bedtools.py --help
pixi run python bin/random_walk_rank_regions_v4.py --help
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

Reference databases are required for annotation and prioritization and must be
downloaded separately using the links below.

| Name | Download |
|---|---|
| hg38 | [download](https://www.6157777.xyz/enhancerinsight/download/enhancer_insight.hg38.tgz) |
| hg19 | [download](https://www.6157777.xyz/enhancerinsight/download/enhancer_insight.hg19.tgz)|
| Pre-built RWR network  | [download](https://www.6157777.xyz/enhancerinsight/download/enhancer_insight.network.tgz) |


After downloading, extract to a directory (referred to as `<refdir>` below):

```bash
tar -xzf enhancerinsight_ref_hg38.tar.gz -C /path/to/refdir/
```

Pre-built RWR network files for prioritization:

| File | Description |
|---|---|
| `prebuilt_network_v2_with_gwas.gpickle` | Base network for Mode 1 and 2A |
| `dbscATAC_edgeprep_compact.base_edges.npz` | Base edge matrix for Mode 2B |
| `dbscATAC_edgeprep_compact.edge_override/<CellType>.npz` | Per-cell-type edge overrides |
| `node_specificity_dbscATAC.nodes.tsv` | Node index for Mode 2B |

Place these files in the same `<refdir>` or a dedicated `bin/` directory and
update the paths in the commands below accordingly.

---

## Usage

For pixi users, prefix every command with `pixi run`. If you are using a manual
installation, run the scripts with your system Python/Rscript directly.

### 1. Annotation

```bash
pixi run python bin/annoEnhancer_bedtools.py \
    -m hg38 \
    -in input.bed \
    -o MyProject \
    -w ./results/
```

**All options:**

| Flag | Default | Required | Description |
|---|---|---|---|
| `-in` | — | **yes** | Input BED file (≥ 3 columns: chr, start, end) |
| `-m` | `hg19` | no | Genome assembly: `hg19`, `hg38`, `mm10`, `mm39` |
| `-o` | `annoEnhancer` | no | Output prefix (project name) |
| `-w` | — | **yes** | Output/working directory |
| `-cb` | — | no | Custom binding-sites BED (adds a `user_binding` Y/N column) |
| `-pri` | `0` | no | Set to `1` to add a database-support priority score column |
| `-t` | — | no | Number of threads (activated when input > 100,000 lines) |
| `-maxt` | `8` | no | Maximum allowed threads |

**Example with all options:**

```bash
pixi run python bin/annoEnhancer_bedtools.py \
    -m hg38 \
    -in regions.bed \
    -o GM12878_H3K27ac \
    -w ./results/ \
    -cb chipseq_peaks.bed \
    -pri 1 \
    -t 8
```

---

### 2. Generating annotation report

After running annotation, generate the interactive HTML report using the
provided R Markdown template:

```bash
pixi run python render_annoEnhancer_report_updated.py \
    --rmd    report-update.Rmd \
    --outdir ./results/ \
    --proj   MyProject \
    --genome hg38 \
    --out    ./results/MyProject_report.html
```

Alternatively, render directly from R:

```r
rmarkdown::render(
  "report-update.Rmd",
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

### 3. Prioritization — Mode 1: trait-guided

Enhancers are prioritized based on their network proximity to trait-associated signals. A selected disease or trait defines seed nodes; these are propagated via RWR. No input score required.

```bash
pixi run python bin/random_walk_rank_regions_v4.py \
    --network     /path/to/prebuilt_network_v2_with_gwas.gpickle \
    --disease-trait "Atrial fibrillation" \
    --input-regions input.bed \
    --restart 0.5 \
    --tol 1e-7 \
    --max-iter 100 \
    --combine-overlaps max \
    --output ranked_output.tsv
```

**All options:**

| Flag | Default | Required | Description |
|---|---|---|---|
| `--network` | — | **yes** | Pre-built network file (.gpickle) |
| `--disease-trait` | — | **yes** | GWAS trait keyword to use as seed (must match trait name in network) |
| `--input-regions` | — | **yes** | Input BED file |
| `--restart` | `0.5` | no | RWR restart probability (0–1). Higher = stronger return to seed nodes |
| `--tol` | `1e-7` | no | Convergence tolerance |
| `--max-iter` | `100` | no | Maximum RWR iterations |
| `--combine-overlaps` | `max` | no | How to combine scores when a region overlaps multiple nodes: `max`, `mean`, `sum` |
| `--output` | — | **yes** | Output TSV file path |

---

### 4. Prioritization — Mode 2A: input signal-guided

Each input region's score (e.g., GWAS −log₁₀(p-value)) is used as initial node heat. Signal is diffused through the network for context-aware ranking.

```bash
pixi run python bin/random_walk_rank_regions_v4.py \
    --network     /path/to/prebuilt_network_v2_with_gwas.gpickle \
    --input-regions input_with_scores.bed \
    --restart 0.3 \
    --tol 1e-7 \
    --max-iter 100 \
    --combine-overlaps max \
    --output ranked_output.tsv
```

The input BED should have a score in column 5 (no `--disease-trait` flag):

```
chr1    713441    714434    region_001    24.15
chr2    208245102 208246500 region_002    29.00
chr8    127742018 127744200 region_003    13.70
```

> **Note:** A restart probability of `0.3` (vs. `0.5` for Mode 1) is recommended
> to allow broader diffusion from sparse input seed nodes.

---

### 5. Prioritization — Mode 2B: cell-type specific

A selected cell type is used to reweight the base network using scATAC-seq chromatin accessibility. Signal propagation identifies enhancers active in the chosen cellular context. Available cell types correspond to the `.npz` files in the
`edge_override/` directory.

```bash
pixi run python bin/random_walk_rank_regions_dynamic.py \
    --base-edges   /path/to/dbscATAC_edgeprep_compact.base_edges.npz \
    --edge-override /path/to/dbscATAC_edgeprep_compact.edge_override/Cardiomyocyte.npz \
    --nodes-index  /path/to/node_specificity_dbscATAC.nodes.tsv \
    --input-regions input.bed \
    --restart 0.5 \
    --combine-overlaps max \
    --output ranked_cardiomyocyte.tsv
```

**All options:**

| Flag | Required | Description |
|---|---|---|
| `--base-edges` | **yes** | Base regulatory network edge matrix (.npz) |
| `--edge-override` | **yes** | Cell-type-specific edge weight matrix (.npz) |
| `--nodes-index` | **yes** | Node ID ↔ index mapping (.tsv) |
| `--input-regions` | **yes** | Input BED file (score in col 5 used if present) |
| `--restart` | no | RWR restart probability (default: `0.5`) |
| `--combine-overlaps` | no | Score combination strategy (default: `max`) |
| `--output` | **yes** | Output TSV file path |

**List available cell types:**

```bash
ls /path/to/dbscATAC_edgeprep_compact.edge_override/*.npz | xargs -n1 basename | sed 's/.npz//'
```

---

## Output files

### Annotation

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

### Annotation output columns

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

| Column | Mode | Description |
|---|---|---|
| `input_region` | all | Region identifier (`chr:start-end`) |
| `rwr_score` | all | RWR score after convergence [0–1] |
| `input_score` | 2A | User-supplied score from BED column 5 |
| `n_overlapped_nodes` | all | Number of network nodes overlapping the region |
| `overlapped_nodes` | all | Semicolon-separated overlapping node coordinates |
| `overlapped_node_scores` | 2B | Per-node RWR score |
| `celltype` | 2B | Selected cell type |
| `all_tiny_components` | 1/2A | `True` if all overlapping nodes are isolated (score unreliable) |
| `has_other_scored_node_in_component` | 1/2A | Network context quality flag |

---

## Input format

Standard **BED format** (tab-separated). Columns 1–3 are required; coordinates
follow the 0-based, half-open convention used by UCSC and bedtools.

```
# Minimum (BED3)
chr1    713441    714434
chr8    127742018 127744200

# With name and score (BED5) — score column used in Mode 2A
chr1    713441    714434    region_001    24.15
chr8    127742018 127744200 region_002    13.70
```

> A region spanning bases 1000–2000 (1-based, inclusive) is written as
> `999  2000` in BED (0-based, exclusive end).

---

## Citation

To be published...

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
