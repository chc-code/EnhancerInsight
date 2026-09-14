#!/usr/bin/env python

import sys
import argparse
import pickle
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import pyranges as pr
import scipy.sparse as sp

from enhancerinsight_report_utils_evidence import generate_static_report_outputs


COORD_RE = re.compile(r'^(chr[^:]+):(\d+)-(\d+)$')
POINT_RE = re.compile(r'^(chr[^:]+):(\d+)$')


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
        ends.append(end + 1 if end == start else end)
        nodes.append(n)
    return pd.DataFrame({
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "node": nodes,
    })


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
    return P, A


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


def compute_local_seed_support(A, seed_vec, max_hops=3, hop_decay=0.5):
    """Compute local seed/heat support without degree normalization.

    The support at hop d is obtained by propagating the seed-strength vector
    through the raw weighted adjacency matrix, so a source node does NOT divide
    its support by its total weighted degree as it does in standard RWR.

      hop 1 contribution: A^T s
      hop 2 contribution: hop_decay * (A^T)^2 s
      hop 3 contribution: hop_decay^2 * (A^T)^3 s
      ...

    Self-loop edges are ignored for this local-support term so that direct
    self evidence is not counted as neighborhood support.  The returned vector
    is max-normalized to [0, 1] before combination with the max-normalized RWR
    score.

    Note that multiple weighted walks contribute additively.  This intentionally
    rewards nodes supported by multiple nearby seeds / paths.
    """
    if seed_vec is None or len(seed_vec) == 0 or max_hops <= 0:
        return np.zeros(A.shape[0], dtype=float)
    if hop_decay < 0:
        raise ValueError("--local-support-hop-decay must be >= 0")

    # P was already constructed from A before this function is called, so it is
    # safe to use a local copy with diagonal entries removed.
    A_local = A.copy().tocsr()
    A_local.setdiag(0.0)
    A_local.eliminate_zeros()

    current = np.asarray(seed_vec, dtype=float).reshape(-1)
    support = np.zeros_like(current, dtype=float)

    for hop in range(1, int(max_hops) + 1):
        current = np.asarray(A_local.T @ current).reshape(-1)
        support += (float(hop_decay) ** (hop - 1)) * current

    support[~np.isfinite(support)] = 0.0
    support[support < 0] = 0.0
    return normalize_vec(support)


def combine_rwr_and_local_support(rwr_norm, local_support_norm, eta=0.3):
    """Combine max-normalized RWR and local support using weight eta."""
    eta = float(eta)
    if eta < 0 or eta > 1:
        raise ValueError("--local-support-eta must be between 0 and 1")
    return (1.0 - eta) * np.asarray(rwr_norm, dtype=float) + eta * np.asarray(local_support_norm, dtype=float)


# -----------------------------
# Input regions (with or without scores)
# -----------------------------

def _parse_point_region(s):
    """Parse 'chrX:pos' into (chrom, pos)."""
    if not isinstance(s, str):
        return None, None
    m = POINT_RE.match(s.strip())
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _is_valid_bed_end(start_text, end_text):
    """Return True only when start/end are integers and end >= start."""
    try:
        start = int(start_text)
        end = int(end_text)
    except (TypeError, ValueError):
        return False
    return end >= start


def load_input_regions(path):
    """
    Load interval or SNP-style inputs.

    Coordinate-string formats (parsed per line):
      chr1:1000-2000 [score] [extra ...]
      chr1:1000      [score] [extra ...]

    Tabular formats are classified for the WHOLE file before parsing:
      interval: chr  start  end  [score] [extra ...]
      SNP:      chr  pos    [score] [extra ...]

    If ANY tabular data line has no valid genomic end in column 3
    (missing, non-integer, or < column 2), the entire file is treated as SNP
    format. SNPs use internal input_region IDs chr:pos-pos, but overlap
    coordinates use [pos, pos+1). display_region keeps chr:pos for final output.
    """
    raw_rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_rows.append((line, re.split(r'\s+', line)))

    tabular_rows = [parts for _, parts in raw_rows if ":" not in parts[0]]
    tabular_is_snp = False
    if tabular_rows:
        tabular_is_snp = any(
            len(parts) < 3 or not _is_valid_bed_end(parts[1], parts[2])
            for parts in tabular_rows
        )

    regions = []
    display_regions = []
    chroms, starts, ends = [], [], []
    scores = []
    any_score = False

    for line, parts in raw_rows:
        chrom = start = end = None
        region = display_region = None
        score_val = np.nan

        # chr:start-end [score]
        c, s, e = parse_coord_region(parts[0])
        if c is not None:
            chrom, start, end = c, s, e
            region = parts[0]
            display_region = parts[0]
            if len(parts) >= 2:
                try:
                    score_val = float(parts[1])
                    any_score = True
                except ValueError:
                    score_val = np.nan
        else:
            # chr:pos [score]
            c, pos = _parse_point_region(parts[0])
            if c is not None:
                chrom = c
                start = pos
                end = pos + 1
                region = f"{chrom}:{pos}-{pos}"
                display_region = f"{chrom}:{pos}"
                if len(parts) >= 2:
                    try:
                        score_val = float(parts[1])
                        any_score = True
                    except ValueError:
                        score_val = np.nan

            elif tabular_is_snp and len(parts) >= 2:
                # Whole-file SNP format: chr pos [score]
                try:
                    chrom = parts[0]
                    pos = int(parts[1])
                    start = pos
                    end = pos + 1
                    region = f"{chrom}:{pos}-{pos}"
                    display_region = f"{chrom}:{pos}"
                    if len(parts) >= 3:
                        try:
                            score_val = float(parts[2])
                            any_score = True
                        except ValueError:
                            score_val = np.nan
                except ValueError:
                    chrom = start = end = region = display_region = None

            elif (not tabular_is_snp) and len(parts) >= 3:
                # Whole-file interval format: chr start end [score]
                try:
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    if end < start:
                        raise ValueError
                    region = f"{chrom}:{start}-{end}"
                    display_region = region
                    if len(parts) >= 4:
                        try:
                            score_val = float(parts[3])
                            any_score = True
                        except ValueError:
                            score_val = np.nan
                except ValueError:
                    chrom = start = end = region = display_region = None

        if chrom is None:
            print(f"WARNING: cannot parse input line, skipping: {line}")
            continue

        regions.append(region)
        display_regions.append(display_region)
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
        scores.append(score_val)

    data = {
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "input_region": regions,
        "display_region": display_regions,
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
# Build node heat from input regions
# -----------------------------

def build_node_heat_from_region_scores(G, node_list, node_to_idx,
                                       input_regions_df, combine_mode="max",
                                       return_scored_nodes=False, equal_heat=False):
    """Build the RWR restart vector from region-level input scores.

    Each input region contributes a fixed total amount of heat equal to its
    ``input_score``.  If a region overlaps multiple network nodes, that heat is
    divided equally among its unique overlapped nodes.  Contributions from
    different input regions that overlap the same node are then summed.  The
    resulting node-level heat vector is normalized to sum to one.

    ``combine_mode`` is retained only for backward-compatible calls.  It is not
    used during seed construction; ``--combine-overlaps`` applies only when
    stationary node scores are aggregated back to a region-level final score.

    If ``equal_heat=True`` (input without scores), every node overlapped by an
    input region gets the same initial heat.

    If ``return_scored_nodes=True``, also return the set of nodes receiving
    positive initial heat before normalization.
    """
    if "input_score" not in input_regions_df.columns and not equal_heat:
        print("No input_score column found; cannot build node heat.")
        return (None, set()) if return_scored_nodes else None

    nodes_df = build_nodes_interval_df(G)
    if nodes_df.empty:
        print("WARNING: no coordinate-like nodes in graph; cannot map region scores.")
        return (None, set()) if return_scored_nodes else None

    ov_df = pr.PyRanges(nodes_df).join(pr.PyRanges(input_regions_df)).df
    if ov_df.empty:
        print("WARNING: no overlaps between regions and nodes for scores; cannot build node heat.")
        return (None, set()) if return_scored_nodes else None

    # Use unique region-to-node mappings so duplicated interval joins do not
    # duplicate a region's total initial evidence.
    region_to_nodes = defaultdict(set)
    for row in ov_df.itertuples(index=False):
        region = str(getattr(row, "input_region"))
        node = getattr(row, "node")
        if node in node_to_idx:
            region_to_nodes[region].add(node)

    score_by_region = {}
    for row in input_regions_df.itertuples(index=False):
        region = str(getattr(row, "input_region"))
        score = getattr(row, "input_score", np.nan)
        if pd.isna(score):
            continue
        # input_region is expected to be unique; retain the first finite value
        # if a duplicated row is present.
        score_by_region.setdefault(region, float(score))

    heat = np.zeros(len(node_list), dtype=float)
    scored_nodes = set()
    for region, nodes in region_to_nodes.items():
        if equal_heat:
            for node in nodes:
                idx = node_to_idx.get(node)
                if idx is not None:
                    heat[idx] = 1.0
                    scored_nodes.add(node)
            continue
        score = score_by_region.get(region)
        if score is None or not np.isfinite(score) or len(nodes) == 0:
            continue
        share = float(score) / float(len(nodes))
        for node in nodes:
            idx = node_to_idx.get(node)
            if idx is None:
                continue
            heat[idx] += share
            if share > 0:
                scored_nodes.add(node)

    total = float(heat.sum())
    if total <= 0:
        print("WARNING: node heat sums to zero; cannot build seed vector.")
        return (None, set()) if return_scored_nodes else None

    heat /= total
    if return_scored_nodes:
        return heat, scored_nodes
    return heat


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
            "Context-independent prioritization: random walk with restart on the pre-built network to rank regions.\n"
            "Input scores are used as initial node heat; without scores, all overlapped nodes start with equal heat."
        )
    )
    p.add_argument("--network", required=True,
                   help="Pre-built network .gpickle (genes/regions as chr:start-end).")
    p.add_argument("--input-regions", required=True,
                   help="Input regions file. Supported formats:\n"
                        "  1) chr:start-end [score]\n"
                        "  2) chr start end [score]\n"
                        "  3) chr:pos [score]\n"
                        "  4) chr pos [score]  (SNPs)\n"
                        "Extra columns are ignored.")
    p.add_argument("--restart", type=float, default=0.5,
                   help="Restart probability for RWR. Default=0.5")
    p.add_argument("--tol", type=float, default=1e-7,
                   help="L1 convergence tolerance. Default=1e-7")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Maximum RWR iterations. Default=100")
    p.add_argument("--combine-overlaps", choices=["max", "mean", "union"], default="max",
                   help="How to combine scores when a region overlaps multiple nodes. Default=max")
    p.add_argument("--local-support-eta", type=float, default=0.3,
                   help="Weight of local seed/heat support in the final static-network score. Default=0.3")
    p.add_argument("--local-support-max-hops", type=int, default=3,
                   help="Maximum hops used for local seed/heat support. Default=3")
    p.add_argument("--local-support-hop-decay", type=float, default=0.5,
                   help="Multiplicative decay for each additional local-support hop. Default=0.5")
    p.add_argument("--annotate-top-n", type=int, default=0,
                   help="If >0, add component-aware annotations for the top N ranked regions.")
    p.add_argument("--tiny-component-size", type=int, default=2,
                   help="Component size threshold for tiny components. Default=2")
    p.add_argument("--output", required=True,
                   help="Output directory. The main ranked table will be written to <output>/ranking.tsv, and report-ready files to <output>/report/.")
    p.add_argument("--report-top-n", type=int, default=20,
                   help="Number of top regions to export to report/top20.tsv. Default=20")
    p.add_argument("--report-top-network-n", type=int, default=3,
                   help="Number of top regions for local network export. Default=3")
    p.add_argument("--report-max-neighbors", type=int, default=25,
                   help="Maximum neighbors per overlapped node in local network export. Default=25")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_file = output_dir / "ranking.tsv"
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading network from {args.network}")
    with open(args.network, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build node indexing and transition matrix
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    print("Building transition matrix ...")
    P, A_weighted = build_transition_matrix(G, node_list, node_to_idx)

    print(f"Loading input regions from {args.input_regions}")
    all_regions, input_regions_df, has_scores = load_input_regions(args.input_regions)
    print(f"  Loaded {len(all_regions)} regions; has_scores={has_scores}")

    mode = "input_scores" if has_scores else "input_regions_binary"
    print(f"\nOperating mode: {mode}")

    print("Building node heat from input regions ...")
    s_heat, scored_nodes = build_node_heat_from_region_scores(
        G, node_list, node_to_idx,
        input_regions_df,
        return_scored_nodes=True,
        equal_heat=not has_scores,
    )
    if s_heat is None:
        print("ERROR: could not build node heat from input regions (no overlap with network nodes, or no positive scores).")
        sys.exit(1)

    seed_vector_for_report = s_heat.copy()
    reference_nodes = set(scored_nodes)
    reference_label = "scored" if has_scores else "input"

    print(f"Initial nodes with heat: {len(reference_nodes)}")

    print(f"Running RWR from heat (restart={args.restart}) ...")
    p = random_walk_with_restart(P, s_heat,
                                 restart=args.restart,
                                 tol=args.tol,
                                 max_iter=args.max_iter)
    p_norm = normalize_vec(p)

    # Local heat support uses the raw weighted adjacency (no degree
    # normalization), so multiple nearby high-heat inputs can directly rescue a
    # candidate.
    print(
        f"Computing local heat support (max_hops={args.local_support_max_hops}, "
        f"decay={args.local_support_hop_decay}, eta={args.local_support_eta}) ..."
    )
    local_support_node = compute_local_seed_support(
        A_weighted, s_heat,
        max_hops=args.local_support_max_hops,
        hop_decay=args.local_support_hop_decay,
    )
    final_node_score = combine_rwr_and_local_support(
        p_norm, local_support_node, eta=args.local_support_eta
    )

    print(f"Mapping node scores back to regions (combine={args.combine_overlaps}) ...")
    region_scores_rwr = map_regions_to_nodes(
        G, p_norm, node_to_idx, input_regions_df,
        combine_mode=args.combine_overlaps
    )
    region_scores_local = map_regions_to_nodes(
        G, local_support_node, node_to_idx, input_regions_df,
        combine_mode=args.combine_overlaps
    )
    region_scores = map_regions_to_nodes(
        G, final_node_score, node_to_idx, input_regions_df,
        combine_mode=args.combine_overlaps
    )

    rows = []
    for region in sorted(region_scores.keys()):
        info = region_scores[region]
        raw_score = ""
        subset = input_regions_df[input_regions_df["input_region"] == region]
        if not subset.empty and "input_score" in subset.columns:
            raw_score = subset["input_score"].iloc[0]
        rows.append({
            "input_region": region,
            "rwr_score_base": region_scores_rwr[region]["score"],
            "local_support_score": region_scores_local[region]["score"],
            "rwr_score": info["score"],
            "final_score": info["score"],
            "input_score": raw_score,
            "n_overlapped_nodes": info["n_nodes"],
            "overlapped_nodes": ";".join(info["nodes"]) if info["nodes"] else "",
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df[pd.to_numeric(out_df["n_overlapped_nodes"], errors="coerce").fillna(0) > 0].copy()
    out_df["_input_score_sort"] = pd.to_numeric(out_df.get("input_score"), errors="coerce").fillna(-np.inf)
    out_df = out_df.sort_values(
        ["rwr_score", "_input_score_sort"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns="_input_score_sort").reset_index(drop=True)

    out_df["local_support_eta"] = float(args.local_support_eta)
    out_df["local_support_max_hops"] = int(args.local_support_max_hops)
    out_df["local_support_hop_decay"] = float(args.local_support_hop_decay)

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

    # Restore chr:pos display for SNP inputs only after all internal mapping,
    # annotation, scoring, and ranking have finished.
    display_map = dict(zip(input_regions_df["input_region"], input_regions_df["display_region"]))
    out_df["input_region"] = out_df["input_region"].map(lambda r: display_map.get(r, r))

    print(f"\nWriting ranked regions to {ranking_file}")
    out_df.to_csv(ranking_file, sep="\t", index=False)

    print(f"Writing report-ready tables to {report_dir}")
    generate_static_report_outputs(
        out_df=out_df,
        G=G,
        output_tsv=str(ranking_file),
        mode=mode,
        args=args,
        node_list=node_list,
        node_scores=final_node_score,
        seed_vector=seed_vector_for_report,
        reference_nodes=reference_nodes,
        reference_label=reference_label,
        report_dir=str(report_dir),
        top_n=args.report_top_n,
        top_network_n=args.report_top_network_n,
        max_neighbors_per_node=args.report_max_neighbors,
    )
    print("Done.")


if __name__ == "__main__":
    main()

