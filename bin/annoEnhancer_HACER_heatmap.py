#!/usr/bin/env python3
"""enhanno_hacer_heatmap.py

Python port of `hacer-test.R` (core HACER heatmap logic).

Goal:
- Build an enhancer-by-sample matrix from a long table:
    enhancer_id  sample_id  activity
- Filter rows/columns similarly to the legacy R code.
- Use a tissue mapping file (sample_id -> tissue) to order columns and add
  a column annotation.
- Perform row clustering on a log2-transformed matrix (with zeros replaced by a
  small constant) and plot a heatmap where zeros are shown as NA (gray).

Notes:
- The original R script contains multiple experimental blocks. This Python
  script implements the first "main" block that is typically used for HACER
  tissue/cell-line heatmap visualization.
- To match the R behavior as closely as possible, we:
  * Replace 0 with min_nonzero^2 for clustering-only matrix
  * Replace 0 with NA for the plotted matrix
  * Use log2 transform
  * Use Ward linkage on Euclidean distance for row clustering

Input files:
- --hacer-long: tab-delimited 3 columns (enhancer, sample, activity)
- --tissue-map: tab-delimited 2 columns (sample, tissue); empty tissue rows ignored

Outputs:
- --out-png: PNG heatmap
- --out-matrix: optional TSV of the filtered matrix (wide)
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate HACER-style heatmap (Python port).")
    ap.add_argument("--hacer-long", required=True, help="3-col TSV: enhancer_id sample_id activity")
    ap.add_argument("--tissue-map", required=True, help="2-col TSV: sample_id tissue")
    ap.add_argument("--out-png", required=True, help="Output PNG path")
    ap.add_argument("--out-matrix", default=None, help="Optional output wide matrix TSV")

    # Filters matching legacy defaults
    ap.add_argument("--min-nonzero-per-col", type=int, default=20,
                    help="Keep columns with > this many nonzero entries")
    ap.add_argument("--min-nonzero-per-row", type=int, default=5,
                    help="Keep rows with > this many nonzero entries")
    ap.add_argument("--min-tissue-samples", type=int, default=4,
                    help="Keep tissues with > this many samples")

    # Tissue-specific selection (legacy uses p<1e-4 OR high-coverage rows)
    ap.add_argument("--tissue_p", type=float, default=1e-4,
                    help="Row-level tissue-specific t-test p-value threshold")
    ap.add_argument("--high_coverage_frac", type=float, default=0.8,
                    help="Row passes if nonzero count > frac*ncol")

    # Plot settings
    ap.add_argument("--fig-width", type=float, default=16)
    ap.add_argument("--fig-height", type=float, default=8)
    ap.add_argument("--dpi", type=int, default=200)

    return ap.parse_args()


def read_hacer_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["enhancer", "sample", "activity"], dtype={0: str, 1: str})
    df["activity"] = pd.to_numeric(df["activity"], errors="coerce").fillna(0.0)
    return df


def read_tissue_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["sample", "tissue"], dtype=str)
    df = df.fillna("")
    df = df[df["tissue"] != ""]
    return df


def wide_matrix(df_long: pd.DataFrame) -> pd.DataFrame:
    # Equivalent to the R for-loop filling a 0 matrix
    wide = df_long.pivot_table(index="enhancer", columns="sample", values="activity", aggfunc="max", fill_value=0.0)
    return wide


def filter_rows_cols(mat: pd.DataFrame, min_nz_col: int, min_nz_row: int) -> pd.DataFrame:
    # Keep columns with >min_nz_col nonzero values
    col_keep = (mat.gt(0).sum(axis=0) > min_nz_col)
    mat = mat.loc[:, col_keep]

    # Keep rows with >min_nz_row nonzero values
    row_keep = (mat.gt(0).sum(axis=1) > min_nz_row)
    mat = mat.loc[row_keep, :]
    return mat


def filter_tissues(mat: pd.DataFrame, tissue_map: pd.DataFrame, min_tissue_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep tissues with > min_tissue_samples samples
    counts = tissue_map["tissue"].value_counts()
    keep_tissues = set(counts[counts > min_tissue_samples].index.tolist())
    tissue_map = tissue_map[tissue_map["tissue"].isin(keep_tissues)].copy()

    # Keep only samples present in both
    samples_in_mat = set(mat.columns)
    tissue_map = tissue_map[tissue_map["sample"].isin(samples_in_mat)].copy()

    # Order samples by tissue (like R: cell <- cell[order(cell$V2),])
    tissue_map = tissue_map.sort_values(["tissue", "sample"], kind="mergesort")

    # Subset matrix to ordered samples
    ordered_samples = tissue_map["sample"].tolist()
    mat = mat.loc[:, ordered_samples]

    return mat, tissue_map


def row_selection_tissue_specific(mat: pd.DataFrame, tissue_map: pd.DataFrame, tissue_p: float, high_coverage_frac: float) -> pd.DataFrame:
    # High coverage rows: > frac*ncol nonzero
    ncol = mat.shape[1]
    high_cov = (mat.gt(0).sum(axis=1) > (ncol * high_coverage_frac))

    # Tissue-specific t-tests like the R code:
    # For each row, for each tissue: t.test(values in tissue vs not tissue)
    tissues = tissue_map["tissue"].unique().tolist()
    sample_to_tissue: Dict[str, str] = dict(zip(tissue_map["sample"], tissue_map["tissue"]))

    # Precompute boolean masks per tissue (columns)
    cols = mat.columns.tolist()
    tissue_masks = {t: np.array([sample_to_tissue[c] == t for c in cols], dtype=bool) for t in tissues}

    # Vectorized-ish per row; still loops but acceptable for typical matrix sizes
    from scipy.stats import ttest_ind

    pvals = np.ones(mat.shape[0], dtype=float)
    values = mat.to_numpy(dtype=float)

    for i in range(values.shape[0]):
        row = values[i, :]
        best_p = 1.0
        for t in tissues:
            m = tissue_masks[t]
            t1 = row[m]
            t2 = row[~m]
            if t1.size < 2 or t2.size < 2:
                continue
            # Welch t-test (closest to R's default t.test)
            stat = ttest_ind(t1, t2, equal_var=False, nan_policy="omit")
            p = float(stat.pvalue) if not math.isnan(stat.pvalue) else 1.0
            best_p = min(best_p, p)
            if best_p < tissue_p:
                break
        pvals[i] = best_p

    keep = (pvals < tissue_p) | high_cov.to_numpy()
    return mat.loc[keep, :]


def prepare_for_plot(mat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Determine min nonzero
    arr = mat.to_numpy(dtype=float)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        raise ValueError("Matrix has no nonzero values after filtering.")

    min_nonzero = float(nonzero.min())

    # Cluster matrix: replace 0 with min_nonzero^2 (legacy)
    cluster_arr = arr.copy()
    cluster_arr[cluster_arr == 0] = min_nonzero ** 2

    # Plot matrix: replace 0 with NaN
    plot_arr = arr.copy()
    plot_arr[plot_arr == 0] = np.nan

    # log2
    cluster_log = np.log2(cluster_arr)
    plot_log = np.log2(plot_arr)

    return cluster_log, plot_log


def compute_row_order(cluster_log: np.ndarray) -> np.ndarray:
    # Ward linkage on Euclidean distance
    Z = linkage(cluster_log, method="ward", metric="euclidean")
    order = leaves_list(Z)
    return order


def draw_heatmap(plot_log: np.ndarray, tissue_map: pd.DataFrame, row_order: np.ndarray, out_png: str,
                 fig_w: float, fig_h: float, dpi: int) -> None:
    # Compute color limits from 1% and 99% quantiles (legacy)
    vmin = float(np.nanquantile(plot_log, 0.01))
    vmax = float(np.nanquantile(plot_log, 0.99))

    # Reorder rows
    plot_log = plot_log[row_order, :]

    # Build a simple column color bar based on tissue
    tissues = tissue_map["tissue"].tolist()
    uniq = sorted(set(tissues))

    # Deterministic palette
    import matplotlib
    cmap_tab = plt.get_cmap("tab20")
    tissue_to_color = {t: cmap_tab(i % cmap_tab.N) for i, t in enumerate(uniq)}
    col_colors = np.array([tissue_to_color[t] for t in tissues])

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.15, 0.85], hspace=0.05)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    ax_bar.imshow(col_colors[np.newaxis, :, :], aspect="auto")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    # Heatmap; NaN shown as light gray
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(plot_log, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Enhancers")

    # Column labels (may be many; keep minimal)
    ax.set_xticks(range(plot_log.shape[1]))
    ax.set_xticklabels(tissue_map["sample"].tolist(), rotation=90, fontsize=6)
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log2(activity)")

    # Legend for tissues
    handles = [matplotlib.patches.Patch(color=tissue_to_color[t], label=t) for t in uniq]
    ax_bar.legend(handles=handles, ncol=min(6, len(uniq)), fontsize=7, loc="center")

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    
    df_long = read_hacer_long(args.hacer_long)
    tissue_map = read_tissue_map(args.tissue_map)

    mat = wide_matrix(df_long)

    mat = filter_rows_cols(mat, args.min_nonzero_per_col, args.min_nonzero_per_row)
    mat, tissue_map = filter_tissues(mat, tissue_map, args.min_tissue_samples)

    mat = row_selection_tissue_specific(mat, tissue_map, args.tissue_p, args.high_coverage_frac)

    if args.out_matrix:
        mat.to_csv(args.out_matrix, sep="\t")

    cluster_log, plot_log = prepare_for_plot(mat)
    row_order = compute_row_order(cluster_log)

    draw_heatmap(plot_log, tissue_map, row_order, args.out_png, args.fig_width, args.fig_height, args.dpi)

    print(f"Wrote heatmap PNG: {args.out_png}")
    if args.out_matrix:
        print(f"Wrote filtered matrix TSV: {args.out_matrix}")


if __name__ == "__main__":
    main()

