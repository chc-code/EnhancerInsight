#!/usr/bin/env python

import sys
import argparse
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import pyranges as pr
import scipy.sparse as sp


COORD_RE = re.compile(r'^(chr[^:]+):(\d+)-(\d+)$')


# -----------------------------
# Coordinate helpers
# -----------------------------

def parse_coord_region(s):
    """Parse 'chrX:start-end' into (chrom, start, end)."""
    if not isinstance(s, str):
        return None, None, None
    m = COORD_RE.match(s.strip())
    if not m:
        return None, None, None
    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3))
    return chrom, start, end


def build_nodes_interval_df(G):
    """Build DataFrame of coordinate-like nodes for overlap."""
    chroms, starts, ends, nodes = [], [], [], []
    for n in G.nodes():
        chrom, start, end = parse_coord_region(n)
        if chrom is None:
            continue
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
        nodes.append(n)
    return pd.DataFrame({
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "node": nodes,
    })


# -----------------------------
# Seed selection for Mode 1
# -----------------------------

def select_seeds(G, disease_trait=None, seeds_file=None):
    """
    Return set of seed node IDs based on disease_trait and/or seeds_file.
    """
    seeds = set()

    # Trait-based seeds from GWAS annotations
    if disease_trait:
        pat = disease_trait.lower()
        for n, attrs in G.nodes(data=True):
            traits = attrs.get("gwas_disease_traits")
            if not traits:
                continue
            if pat in str(traits).lower():
                seeds.add(n)

    # Explicit seeds from file
    if seeds_file:
        with open(seeds_file) as f:
            for line in f:
                nid = line.strip()
                if not nid:
                    continue
                if nid in G:
                    seeds.add(nid)

    return seeds


def build_seed_vector(node_list, node_to_idx, seeds):
    """
    Build seed vector s over nodes:
      - if seeds present: uniform over seeds
      - else: uniform over all nodes
    """
    n = len(node_list)
    s = np.zeros(n, dtype=float)
    if seeds:
        valid = [node_to_idx[n] for n in seeds if n in node_to_idx]
        if valid:
            for idx in valid:
                s[idx] = 1.0
            s /= s.sum()
            return s
        else:
            print("WARNING: no seeds matched node IDs, falling back to uniform.")
    # fallback: uniform
    s[:] = 1.0 / n
    return s


# -----------------------------
# Transition matrix & RWR
# -----------------------------

def build_transition_matrix(G, node_list, node_to_idx):
    """
    Build row-stochastic transition matrix P where
      P_ij = w_ij / sum_k w_ik
    """
    n = len(node_list)
    rows, cols, data = [], [], []
    for u, v, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        if w <= 0:
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        # undirected: add both directions
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ A
    return P


def random_walk_with_restart(P, s, restart=0.5, tol=1e-7, max_iter=100):
    """
    RWR:
      p_{t+1} = (1-restart) * P^T p_t + restart * s
    """
    p = s.copy()
    for it in range(max_iter):
        p_new = (1.0 - restart) * (P.T @ p) + restart * s
        diff = np.linalg.norm(p_new - p, 1)
        p = p_new
        if diff < tol:
            break
    return p


def normalize_vec(x):
    m = float(x.max())
    if m <= 0:
        return np.zeros_like(x)
    return x / m


# -----------------------------
# Input regions (with or without scores)
# -----------------------------

def load_input_regions(path):
    """
    Load input regions.

    Supported input formats:

    1) coord format
       chr1:1000-2000
       chr1:1000-2000   5.3
       chr1:1000-2000   5.3   extra1 extra2 ...
       -> only first 2 columns are considered

    2) BED format
       chr1   1000   2000
       chr1   1000   2000   5.3
       chr1   1000   2000   5.3   extra1 extra2 ...
       -> only first 4 columns are considered

    Returns:
      regions: list of region strings
      df: DataFrame with Chromosome, Start, End, input_region, and possibly 'input_score'
      has_scores: True if a usable score column is present
    """
    regions = []
    chroms, starts, ends = [], [], []
    scores = []
    any_score = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = re.split(r'\s+', line)

            chrom, start, end = None, None, None
            region = None
            score_val = np.nan

            # ---------------------------------
            # Format A: chr:start-end [score]
            # only use first 2 columns
            # ---------------------------------
            chrom, start, end = parse_coord_region(parts[0])
            if chrom is not None:
                region = parts[0]

                if len(parts) >= 2:
                    try:
                        score_val = float(parts[1])
                        any_score = True
                    except ValueError:
                        score_val = np.nan

            # ---------------------------------
            # Format B: BED chr start end [score]
            # only use first 4 columns
            # ---------------------------------
            elif len(parts) >= 3:
                try:
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    region = f"{chrom}:{start}-{end}"

                    if len(parts) >= 4:
                        try:
                            score_val = float(parts[3])
                            any_score = True
                        except ValueError:
                            score_val = np.nan

                except ValueError:
                    chrom, start, end = None, None, None
                    region = None

            if chrom is None:
                print(f"WARNING: cannot parse input line, skipping: {line}")
                continue

            regions.append(region)
            chroms.append(chrom)
            starts.append(start)
            ends.append(end)
            scores.append(score_val)

    data = {
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "input_region": regions,
    }

    if any_score:
        data["input_score"] = scores

    df = pd.DataFrame(data)
    has_scores = "input_score" in df.columns and df["input_score"].notna().any()

    return regions, df, has_scores


# -----------------------------
# Map node scores back to regions
# -----------------------------

def map_regions_to_nodes(G, p, node_to_idx, input_regions_df, combine_mode="max"):
    """
    Map node scores p back to input regions using interval overlaps.

    combine_mode:
      - 'max'   : max node score
      - 'mean'  : mean node score
      - 'union' : 1 - prod(1 - p_i)
    """
    nodes_df = build_nodes_interval_df(G)
    nodes_pr = pr.PyRanges(nodes_df)
    regs_pr = pr.PyRanges(input_regions_df)

    ov = nodes_pr.join(regs_pr)
    ov_df = ov.df

    region_to_nodes = defaultdict(list)
    for row in ov_df.itertuples(index=False):
        node = getattr(row, "node")
        region = getattr(row, "input_region")
        region_to_nodes[region].append(node)

    def combine(scores):
        if not scores:
            return 0.0
        if combine_mode == "max":
            return float(max(scores))
        elif combine_mode == "mean":
            return float(np.mean(scores))
        elif combine_mode == "union":
            prod = 1.0
            for s in scores:
                prod *= (1.0 - s)
            return float(1.0 - prod)
        else:
            return float(max(scores))

    result = {}
    for region in input_regions_df["input_region"].unique():
        nodes = region_to_nodes.get(region, [])
        scores = []
        for n in nodes:
            idx = node_to_idx.get(n)
            if idx is not None:
                scores.append(float(p[idx]))
        score = combine(scores)
        result[region] = {
            "score": score,
            "n_nodes": len(nodes),
            "nodes": nodes
        }

    return result


# -----------------------------
# Build node heat from region scores (Mode 2)
# -----------------------------

def build_node_heat_from_region_scores(G, node_list, node_to_idx,
                                       input_regions_df, combine_mode="max",
                                       return_scored_nodes=False):
    """
    Mode 2: Use region scores (second column) as 'heat',
    diffuse via RWR.

    Steps:
      1) Overlap scored regions with nodes.
      2) Aggregate region scores per node using combine_mode.
      3) Normalize node heat to sum 1 to form seed vector.

    If return_scored_nodes=True, also return the set of nodes that received
    non-zero / non-NaN initial heat before normalization.
    """
    if "input_score" not in input_regions_df.columns:
        print("No input_score column found; cannot build node heat.")
        return (None, set()) if return_scored_nodes else None

    nodes_df = build_nodes_interval_df(G)
    if nodes_df.empty:
        print("WARNING: no coordinate-like nodes in graph; cannot map region scores.")
        return (None, set()) if return_scored_nodes else None

    nodes_pr = pr.PyRanges(nodes_df)
    regs_pr = pr.PyRanges(input_regions_df)

    ov = nodes_pr.join(regs_pr)
    ov_df = ov.df

    if ov_df.empty:
        print("WARNING: no overlaps between regions and nodes for scores; cannot build node heat.")
        return (None, set()) if return_scored_nodes else None

    node_to_scores = defaultdict(list)
    for row in ov_df.itertuples(index=False):
        node = getattr(row, "node")
        s = getattr(row, "input_score", np.nan)
        if pd.isna(s):
            continue
        node_to_scores[node].append(float(s))

    if not node_to_scores:
        print("WARNING: all overlapping scores are NaN; cannot build node heat.")
        return (None, set()) if return_scored_nodes else None

    def combine(scores):
        if not scores:
            return 0.0
        if combine_mode == "max":
            return float(max(scores))
        elif combine_mode == "mean":
            return float(np.mean(scores))
        elif combine_mode == "union":
            arr = np.array(scores)
            m = arr.max()
            if m <= 0:
                return 0.0
            arr = arr / m
            prod = 1.0
            for s in arr:
                prod *= (1.0 - s)
            return float(1.0 - prod)
        else:
            return float(max(scores))

    n = len(node_list)
    s = np.zeros(n, dtype=float)
    scored_nodes = set()

    for node, scs in node_to_scores.items():
        idx = node_to_idx.get(node)
        if idx is None:
            continue
        val = combine(scs)
        s[idx] = val
        if val > 0:
            scored_nodes.add(node)

    total = s.sum()
    if total <= 0:
        print("WARNING: node heat sums to zero; cannot build seed vector.")
        return (None, set()) if return_scored_nodes else None

    s /= total

    if return_scored_nodes:
        return s, scored_nodes
    return s


# -----------------------------
# Top-N component-aware annotation
# -----------------------------

def get_top_n_overlapped_nodes(out_df, top_n):
    """
    Collect overlapped nodes from the top N ranked rows.
    """
    if top_n <= 0 or out_df.empty:
        return set()

    top_df = out_df.head(top_n).copy()
    nodes = set()

    for s in top_df["overlapped_nodes"].fillna(""):
        if not s:
            continue
        for x in str(s).split(";"):
            x = x.strip()
            if x:
                nodes.add(x)

    return nodes


def iter_components(G):
    """
    Iterate connected components for Graph / weakly connected for DiGraph.
    """
    if G.is_directed():
        return nx.weakly_connected_components(G)
    return nx.connected_components(G)


def build_component_annotation_maps(G, target_nodes, reference_nodes):
    """
    For target_nodes only, build:
      - component size
      - reference node count in that component

    reference_nodes:
      - heat mode: scored nodes
      - trait mode: seed nodes
    """
    target_nodes = set(target_nodes)
    reference_nodes = set(reference_nodes)

    node_to_component_size = {}
    node_to_reference_count = {}

    if not target_nodes:
        return node_to_component_size, node_to_reference_count

    remaining = set(target_nodes)

    for comp in iter_components(G):
        hit = comp & remaining
        if not hit:
            continue

        comp_size = len(comp)
        ref_count = len(comp & reference_nodes)

        for n in hit:
            node_to_component_size[n] = comp_size
            node_to_reference_count[n] = ref_count

        remaining -= hit
        if not remaining:
            break

    return node_to_component_size, node_to_reference_count


def annotate_top_n_results(out_df, G, top_n, reference_nodes,
                           reference_label="reference",
                           tiny_component_size=2):
    """
    Add component-aware annotations to top N rows only.

    Added columns:
      - component_sizes
      - min_component_size
      - max_component_size
      - any_tiny_component
      - all_tiny_components
      - <reference_label>_node_counts_in_component
      - has_other_<reference_label>_node_in_component
      - max_other_<reference_label>_node_count
    """
    if top_n <= 0 or out_df.empty:
        return out_df

    target_nodes = get_top_n_overlapped_nodes(out_df, top_n)

    node_to_component_size, node_to_ref_count = build_component_annotation_maps(
        G, target_nodes, reference_nodes
    )

    out_df = out_df.copy()

    comp_sizes_col = []
    min_comp_col = []
    max_comp_col = []
    any_tiny_col = []
    all_tiny_col = []
    ref_counts_col = []
    has_other_ref_col = []
    max_other_ref_col = []

    top_regions = set(out_df.head(top_n)["input_region"].tolist())

    for row in out_df.itertuples(index=False):
        region = getattr(row, "input_region")
        overlapped = getattr(row, "overlapped_nodes", "")

        if region not in top_regions or not overlapped:
            comp_sizes_col.append("")
            min_comp_col.append("")
            max_comp_col.append("")
            any_tiny_col.append("")
            all_tiny_col.append("")
            ref_counts_col.append("")
            has_other_ref_col.append("")
            max_other_ref_col.append("")
            continue

        nodes = [x for x in str(overlapped).split(";") if x]

        comp_sizes = []
        ref_counts = []
        other_ref_counts = []

        for n in nodes:
            csize = node_to_component_size.get(n, np.nan)
            rcount = node_to_ref_count.get(n, np.nan)

            comp_sizes.append(csize)
            ref_counts.append(rcount)

            if pd.isna(rcount):
                other_ref_counts.append(np.nan)
            else:
                self_is_ref = 1 if n in reference_nodes else 0
                other_ref_counts.append(int(rcount) - self_is_ref)

        finite_comp_sizes = [int(x) for x in comp_sizes if not pd.isna(x)]
        finite_ref_counts = [int(x) for x in ref_counts if not pd.isna(x)]
        finite_other_ref_counts = [int(x) for x in other_ref_counts if not pd.isna(x)]

        comp_sizes_col.append(";".join(map(str, finite_comp_sizes)) if finite_comp_sizes else "")
        min_comp_col.append(min(finite_comp_sizes) if finite_comp_sizes else "")
        max_comp_col.append(max(finite_comp_sizes) if finite_comp_sizes else "")
        any_tiny_col.append(any(x <= tiny_component_size for x in finite_comp_sizes) if finite_comp_sizes else "")
        all_tiny_col.append(all(x <= tiny_component_size for x in finite_comp_sizes) if finite_comp_sizes else "")
        ref_counts_col.append(";".join(map(str, finite_ref_counts)) if finite_ref_counts else "")
        has_other_ref_col.append(any(x > 0 for x in finite_other_ref_counts) if finite_other_ref_counts else "")
        max_other_ref_col.append(max(finite_other_ref_counts) if finite_other_ref_counts else "")

    out_df["component_sizes"] = comp_sizes_col
    out_df["min_component_size"] = min_comp_col
    out_df["max_component_size"] = max_comp_col
    out_df["any_tiny_component"] = any_tiny_col
    out_df["all_tiny_components"] = all_tiny_col
    out_df[f"{reference_label}_node_counts_in_component"] = ref_counts_col
    out_df[f"has_other_{reference_label}_node_in_component"] = has_other_ref_col
    out_df[f"max_other_{reference_label}_node_count"] = max_other_ref_col

    return out_df


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Random walk with restart on pre-built network to rank regions.\n"
            "Mode 1: if --disease-trait or --seeds-file is provided, treat as trait/seed-based RWR.\n"
            "Mode 2: if no seeds provided and input-regions has scores in 2nd/4th column, "
            "treat scores as heat and diffuse them (HotNet2-like)."
        )
    )
    p.add_argument("--network", required=True,
                   help="Pre-built network .gpickle (genes/regions as chr:start-end).")
    p.add_argument("--input-regions", required=True,
                   help="Input regions file. Supported formats:\n"
                        "  1) chr:start-end [score]\n"
                        "  2) chr start end [score]\n"
                        "Extra columns are ignored.")
    p.add_argument("--disease-trait", default=None,
                   help="Substring to match in gwas_disease_traits for seed selection (Mode 1).")
    p.add_argument("--seeds-file", default=None,
                   help="File with one node ID per line as seeds (Mode 1).")
    p.add_argument("--restart", type=float, default=0.5,
                   help="Restart probability for RWR. Default=0.5")
    p.add_argument("--tol", type=float, default=1e-7,
                   help="L1 convergence tolerance. Default=1e-7")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Maximum RWR iterations. Default=100")
    p.add_argument("--combine-overlaps", choices=["max", "mean", "union"], default="max",
                   help="How to combine scores when a region overlaps multiple nodes. Default=max")
    p.add_argument("--annotate-top-n", type=int, default=0,
                   help="If >0, add component-aware annotations for the top N ranked regions.")
    p.add_argument("--tiny-component-size", type=int, default=2,
                   help="Component size threshold for tiny components. Default=2")
    p.add_argument("--output", required=True,
                   help="Output TSV with ranked regions.")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    print(f"Loading network from {args.network}")
    with open(args.network, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build node indexing and transition matrix
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    print("Building transition matrix ...")
    P = build_transition_matrix(G, node_list, node_to_idx)

    print(f"Loading input regions from {args.input_regions}")
    all_regions, input_regions_df, has_scores = load_input_regions(args.input_regions)
    print(f"  Loaded {len(all_regions)} regions; has_scores={has_scores}")

    # Decide mode
    has_seed_info = bool(args.disease_trait) or bool(args.seeds_file)

    if has_seed_info:
        mode = "trait"
    elif has_scores:
        mode = "heat"
    else:
        mode = "trait_fallback"

    print(f"\nOperating mode: {mode}")

    reference_nodes = set()
    reference_label = "reference"

    # -------------------
    # Mode 1: trait/seed-based RWR
    # -------------------
    if mode in ["trait", "trait_fallback"]:
        seeds = select_seeds(G, disease_trait=args.disease_trait,
                             seeds_file=args.seeds_file)
        reference_nodes = set(seeds)
        reference_label = "seed"

        print(f"Trait/seeds: {len(seeds)}")
        if len(seeds) == 0:
            print("No seeds found; using uniform over all nodes (global centrality).")

        s = build_seed_vector(node_list, node_to_idx, seeds)

        print(f"Running RWR (restart={args.restart}) ...")
        p = random_walk_with_restart(P, s,
                                     restart=args.restart,
                                     tol=args.tol,
                                     max_iter=args.max_iter)
        p_norm = normalize_vec(p)

        print(f"Mapping node scores back to regions (combine={args.combine_overlaps}) ...")
        region_scores = map_regions_to_nodes(G, p_norm, node_to_idx,
                                             input_regions_df,
                                             combine_mode=args.combine_overlaps)

        rows = []
        for region in sorted(region_scores.keys()):
            info = region_scores[region]
            raw_score = ""
            if "input_score" in input_regions_df.columns:
                subset = input_regions_df[input_regions_df["input_region"] == region]
                if not subset.empty:
                    raw_score = subset["input_score"].iloc[0]
            rows.append({
                "input_region": region,
                "rwr_score": info["score"],
                "input_score": raw_score,
                "n_overlapped_nodes": info["n_nodes"],
                "overlapped_nodes": ";".join(info["nodes"]) if info["nodes"] else "",
            })

        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values("rwr_score", ascending=False)

    # -------------------
    # Mode 2: heat-based (HotNet2-like)
    # -------------------
    else:
        print("Building node heat from region scores ...")
        s_heat, scored_nodes = build_node_heat_from_region_scores(
            G, node_list, node_to_idx,
            input_regions_df,
            combine_mode=args.combine_overlaps,
            return_scored_nodes=True
        )
        if s_heat is None:
            print("ERROR: could not build node heat from region scores and no seeds provided.")
            sys.exit(1)

        reference_nodes = set(scored_nodes)
        reference_label = "scored"

        print(f"Initial scored nodes in heat mode: {len(reference_nodes)}")

        print(f"Running RWR from heat (restart={args.restart}) ...")
        p = random_walk_with_restart(P, s_heat,
                                     restart=args.restart,
                                     tol=args.tol,
                                     max_iter=args.max_iter)
        p_norm = normalize_vec(p)

        print(f"Mapping node scores back to regions (combine={args.combine_overlaps}) ...")
        region_scores = map_regions_to_nodes(G, p_norm, node_to_idx,
                                             input_regions_df,
                                             combine_mode=args.combine_overlaps)

        rows = []
        for region in sorted(region_scores.keys()):
            info = region_scores[region]
            raw_score = ""
            subset = input_regions_df[input_regions_df["input_region"] == region]
            if not subset.empty and "input_score" in subset.columns:
                raw_score = subset["input_score"].iloc[0]
            rows.append({
                "input_region": region,
                "rwr_score": info["score"],
                "input_score": raw_score,
                "n_overlapped_nodes": info["n_nodes"],
                "overlapped_nodes": ";".join(info["nodes"]) if info["nodes"] else "",
            })

        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values("rwr_score", ascending=False)

    # -------------------
    # Optional top-N annotation
    # -------------------
    if args.annotate_top_n > 0:
        print(f"\nAnnotating top {args.annotate_top_n} regions with component-aware summaries ...")
        out_df = annotate_top_n_results(
            out_df,
            G,
            top_n=args.annotate_top_n,
            reference_nodes=reference_nodes,
            reference_label=reference_label,
            tiny_component_size=args.tiny_component_size
        )

    print(f"\nWriting ranked regions to {args.output}")
    out_df.to_csv(args.output, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    main()

