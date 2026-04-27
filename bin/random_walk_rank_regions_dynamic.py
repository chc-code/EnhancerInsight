#!/usr/bin/env python3

import argparse
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyranges as pr


COORD_RE = re.compile(r'^(chr[^:]+):(\d+)-(\d+)$')


# =========================
# Coordinate helpers
# =========================

def parse_coord_region(s):
    if not isinstance(s, str):
        return None, None, None
    m = COORD_RE.match(s.strip())
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def build_nodes_coord_df(nodes):
    chroms, starts, ends, node_ids, idxs = [], [], [], [], []
    for i, n in enumerate(nodes):
        c, s, e = parse_coord_region(n)
        if c is None:
            continue
        chroms.append(c)
        starts.append(s)
        ends.append(e)
        node_ids.append(n)
        idxs.append(i)
    return pd.DataFrame({
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "node_id": node_ids,
        "node_idx": idxs,
    })


# =========================
# Input regions
# =========================

def load_input_regions(path):
    """
    Supported formats:
      1) chr:start-end
      2) chr:start-end   score
      3) chr start end
      4) chr start end   score

    Extra columns are ignored.
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

            parts = re.split(r"\s+", line)

            chrom, start, end = None, None, None
            region = None
            score_val = np.nan

            c, s, e = parse_coord_region(parts[0])
            if c is not None:
                chrom, start, end = c, s, e
                region = parts[0]
                if len(parts) >= 2:
                    try:
                        score_val = float(parts[1])
                        any_score = True
                    except ValueError:
                        score_val = np.nan
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
                    chrom, start, end, region = None, None, None, None

            if chrom is None:
                print(f"WARNING: cannot parse line, skipped: {line}", file=sys.stderr)
                continue

            regions.append(region)
            chroms.append(chrom)
            starts.append(start)
            ends.append(end)
            scores.append(score_val)

    df = pd.DataFrame({
        "input_region": regions,
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
    })
    if any_score:
        df["input_score"] = scores

    has_scores = "input_score" in df.columns and df["input_score"].notna().any()
    return df, has_scores


# =========================
# Node / graph loading
# =========================

def load_node_index(path):
    df = pd.read_csv(path, sep="\t")
    if "row_index" not in df.columns or "node_id" not in df.columns:
        raise ValueError(f"{path} must contain columns: row_index, node_id")
    df = df.sort_values("row_index")
    nodes = df["node_id"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    return nodes, node_to_idx


def load_base_edges(path):
    z = np.load(path)
    u_idx = z["u_idx"].astype(np.int32, copy=False)
    v_idx = z["v_idx"].astype(np.int32, copy=False)
    base_weight = z["base_weight"].astype(np.float32, copy=False)
    return u_idx, v_idx, base_weight


def load_override(path):
    z = np.load(path, allow_pickle=True)
    edge_idx = z["edge_idx"].astype(np.int32, copy=False)
    modifier = z["modifier"].astype(np.float32, copy=False)
    default_modifier = float(z["default_modifier"][0])
    floor_ct = float(z["floor_ct"][0])
    celltype = str(z["celltype"][0]) if "celltype" in z else "unknown"
    return edge_idx, modifier, default_modifier, floor_ct, celltype


# =========================
# Overlap mapping
# =========================

def build_region_to_nodes_map(regions_df, nodes_coord_df):
    if nodes_coord_df.empty:
        return defaultdict(list), defaultdict(list)

    regs_pr = pr.PyRanges(regions_df[["Chromosome", "Start", "End", "input_region"]])
    nodes_pr = pr.PyRanges(nodes_coord_df[["Chromosome", "Start", "End", "node_id", "node_idx"]])

    ov = regs_pr.join(nodes_pr)
    ov_df = ov.df

    reg_to_nodes_idx = defaultdict(list)
    reg_to_nodes_id = defaultdict(list)

    for row in ov_df.itertuples(index=False):
        r = getattr(row, "input_region")
        reg_to_nodes_idx[r].append(getattr(row, "node_idx"))
        reg_to_nodes_id[r].append(getattr(row, "node_id"))

    return reg_to_nodes_idx, reg_to_nodes_id


# =========================
# Seed selection
# =========================

def select_seeds_from_network(G, disease_trait=None, seeds_file=None):
    seeds = set()

    if disease_trait:
        pat = disease_trait.lower()
        for n, attrs in G.nodes(data=True):
            traits = attrs.get("gwas_disease_traits")
            if traits and pat in str(traits).lower():
                seeds.add(n)

    if seeds_file:
        with open(seeds_file) as f:
            for line in f:
                nid = line.strip()
                if nid:
                    seeds.add(nid)

    return seeds


def build_seed_vector_from_seed_nodes(nodes, node_to_idx, seeds):
    n = len(nodes)
    s = np.zeros(n, dtype=np.float64)
    valid = [node_to_idx[x] for x in seeds if x in node_to_idx]
    if not valid:
        return None
    s[valid] = 1.0
    s /= s.sum()
    return s


def build_seed_vector_from_region_scores(regions_df, reg_to_nodes_idx, n_nodes, combine_mode="max"):
    if "input_score" not in regions_df.columns:
        return None, set()

    node_to_scores = defaultdict(list)

    for row in regions_df.itertuples(index=False):
        region = getattr(row, "input_region")
        score = getattr(row, "input_score", np.nan)
        if pd.isna(score):
            continue
        idxs = reg_to_nodes_idx.get(region, [])
        for idx in idxs:
            node_to_scores[idx].append(float(score))

    if not node_to_scores:
        return None, set()

    def combine(vals):
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            return 0.0
        if combine_mode == "mean":
            return float(arr.mean())
        if combine_mode == "union":
            # normalize to [0,1] if raw scores are not already bounded
            m = arr.max()
            if m <= 0:
                return 0.0
            arr = arr / m
            arr = np.clip(arr, 0, 1)
            return float(1.0 - np.prod(1.0 - arr))
        return float(arr.max())

    s = np.zeros(n_nodes, dtype=np.float64)
    scored_nodes = set()

    for idx, vals in node_to_scores.items():
        val = combine(vals)
        s[idx] = val
        if val > 0:
            scored_nodes.add(idx)

    total = s.sum()
    if total <= 0:
        return None, set()

    s /= total
    return s, scored_nodes


def build_seed_vector_from_input_regions_binary(reg_to_nodes_idx, region_ids, n_nodes):
    s = np.zeros(n_nodes, dtype=np.float64)
    seen = set()
    for r in region_ids:
        for idx in reg_to_nodes_idx.get(r, []):
            seen.add(idx)
    if not seen:
        return None, set()
    idxs = sorted(seen)
    s[idxs] = 1.0
    s /= s.sum()
    return s, set(idxs)


# =========================
# Dynamic graph
# =========================

def reconstruct_dynamic_edge_weights(base_weight, edge_idx, modifier, default_modifier):
    w = base_weight.astype(np.float32, copy=True)
    w *= np.float32(default_modifier)
    if len(edge_idx) > 0:
        w[edge_idx] = base_weight[edge_idx] * modifier
    return w


def compute_weighted_degree(n_nodes, u_idx, v_idx, w):
    deg = np.bincount(u_idx, weights=w, minlength=n_nodes).astype(np.float64)
    deg += np.bincount(v_idx, weights=w, minlength=n_nodes).astype(np.float64)
    deg[deg == 0] = 1.0
    return deg


# =========================
# Matrix-free RWR
# =========================

def random_walk_with_restart_edge_arrays(
    n_nodes,
    u_idx,
    v_idx,
    w,
    seed_vec,
    restart=0.5,
    tol=1e-9,
    max_iter=100,
    verbose=False,
):
    """
    P is row-stochastic on an undirected weighted graph:
      P_ij = w_ij / sum_k w_ik

    We iterate:
      p_{t+1} = (1-r) * P^T p_t + r * seed

    Without materializing sparse P.
    """
    deg = compute_weighted_degree(n_nodes, u_idx, v_idx, w)
    p = seed_vec.astype(np.float64, copy=True)

    for it in range(max_iter):
        # contribution from u -> v and v -> u
        contrib_u_to_v = p[u_idx] * w / deg[u_idx]
        contrib_v_to_u = p[v_idx] * w / deg[v_idx]

        q = np.bincount(v_idx, weights=contrib_u_to_v, minlength=n_nodes).astype(np.float64)
        q += np.bincount(u_idx, weights=contrib_v_to_u, minlength=n_nodes).astype(np.float64)

        p_next = (1.0 - restart) * q + restart * seed_vec
        diff = np.abs(p_next - p).sum()

        if verbose:
            print(f"  Iter {it+1}, L1 diff = {diff:.3e}", file=sys.stderr)

        p = p_next
        if diff < tol:
            if verbose:
                print(f"Converged after {it+1} iterations.", file=sys.stderr)
            break

    p[p < 0] = 0
    s = p.sum()
    if s > 0:
        p /= s
    return p


# =========================
# Region aggregation
# =========================

def combine_scores(scores, mode="max"):
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    if mode == "mean":
        return float(arr.mean())
    if mode == "union":
        arr = np.clip(arr, 0, 1)
        return float(1.0 - np.prod(1.0 - arr))
    return float(arr.max())


def map_scores_back_to_regions(region_ids, reg_to_nodes_idx, reg_to_nodes_id, p, combine_mode="max"):
    rows = []
    for r in region_ids:
        idxs = reg_to_nodes_idx.get(r, [])
        ids = reg_to_nodes_id.get(r, [])

        if not idxs:
            rows.append({
                "input_region": r,
                "rwr_score": 0.0,
                "n_overlapped_nodes": 0,
                "overlapped_nodes": "",
                "overlapped_node_scores": "",
            })
            continue

        scores = [float(p[i]) for i in idxs]
        agg = combine_scores(scores, combine_mode)

        rows.append({
            "input_region": r,
            "rwr_score": agg,
            "n_overlapped_nodes": len(idxs),
            "overlapped_nodes": ";".join(ids),
            "overlapped_node_scores": ";".join(f"{x:.6g}" for x in scores),
        })

    return pd.DataFrame(rows)


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Dynamic-edge RWR for region ranking using compact cell-type-specific edge encoding."
    )

    ap.add_argument("--base-edges", required=True,
                    help="base_edges.npz from compact edge preprocessing")
    ap.add_argument("--edge-override", required=True,
                    help="One celltype override .npz from compact edge preprocessing")
    ap.add_argument("--nodes-index", required=True,
                    help="nodes.tsv with row_index,node_id")
    ap.add_argument("--input-regions", required=True)

    ap.add_argument("--network", default=None,
                    help="Optional .gpickle; required only if using --disease-trait")
    ap.add_argument("--disease-trait", default=None,
                    help="Substring match on node attribute gwas_disease_traits")
    ap.add_argument("--seeds-file", default=None,
                    help="One node ID per line")

    ap.add_argument("--restart", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=100)

    ap.add_argument("--combine-overlaps",
                    choices=["max", "mean", "union"],
                    default="max")

    ap.add_argument("--output", required=True)
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


# =========================
# Main
# =========================

def main():
    args = parse_args()

    print(f"Loading nodes index: {args.nodes_index}", file=sys.stderr)
    nodes, node_to_idx = load_node_index(args.nodes_index)
    n_nodes = len(nodes)
    print(f"  n_nodes = {n_nodes}", file=sys.stderr)

    print(f"Loading base edges: {args.base_edges}", file=sys.stderr)
    u_idx, v_idx, base_weight = load_base_edges(args.base_edges)
    print(f"  n_edges = {len(u_idx)}", file=sys.stderr)

    print(f"Loading edge override: {args.edge_override}", file=sys.stderr)
    edge_idx, modifier, default_modifier, floor_ct, celltype = load_override(args.edge_override)
    print(f"  celltype = {celltype}", file=sys.stderr)
    print(f"  floor_ct = {floor_ct:.6g}", file=sys.stderr)
    print(f"  default_modifier = {default_modifier:.6g}", file=sys.stderr)
    print(f"  override edges = {len(edge_idx)}", file=sys.stderr)

    print(f"Loading input regions: {args.input_regions}", file=sys.stderr)
    regions_df, has_scores = load_input_regions(args.input_regions)
    region_ids = regions_df["input_region"].tolist()
    print(f"  input regions = {len(region_ids)}, has_scores = {has_scores}", file=sys.stderr)

    print("Building node coordinate table ...", file=sys.stderr)
    nodes_coord_df = build_nodes_coord_df(nodes)
    print(f"  coordinate-like nodes = {len(nodes_coord_df)}", file=sys.stderr)

    print("Building region-to-node overlap map ...", file=sys.stderr)
    reg_to_nodes_idx, reg_to_nodes_id = build_region_to_nodes_map(regions_df, nodes_coord_df)

    # Decide seed mode
    has_seed_info = bool(args.disease_trait) or bool(args.seeds_file)

    seed_vec = None
    reference_nodes_idx = set()
    seed_mode = None

    if has_seed_info:
        if args.disease_trait and not args.network:
            raise ValueError("--network is required when using --disease-trait")

        seeds = set()
        if args.network:
            print(f"Loading network for seed selection: {args.network}", file=sys.stderr)
            with open(args.network, "rb") as f:
                G = pickle.load(f)
            seeds = select_seeds_from_network(
                G,
                disease_trait=args.disease_trait,
                seeds_file=args.seeds_file
            )
        else:
            seeds = select_seeds_from_network(
                G=None,
                disease_trait=None,
                seeds_file=args.seeds_file
            )

        seed_vec = build_seed_vector_from_seed_nodes(nodes, node_to_idx, seeds)
        if seed_vec is None:
            raise ValueError("No valid seeds matched nodes.tsv")
        reference_nodes_idx = {node_to_idx[x] for x in seeds if x in node_to_idx}
        seed_mode = "seed_nodes"

    elif has_scores:
        seed_vec, reference_nodes_idx = build_seed_vector_from_region_scores(
            regions_df,
            reg_to_nodes_idx,
            n_nodes,
            combine_mode=args.combine_overlaps
        )
        if seed_vec is None:
            raise ValueError("Could not build seed vector from input region scores")
        seed_mode = "input_scores"

    else:
        seed_vec, reference_nodes_idx = build_seed_vector_from_input_regions_binary(
            reg_to_nodes_idx,
            region_ids,
            n_nodes
        )
        if seed_vec is None:
            raise ValueError("Input regions do not overlap any network nodes")
        seed_mode = "input_regions_binary"

    print(f"Seed mode = {seed_mode}", file=sys.stderr)
    print(f"  seed support nodes = {len(reference_nodes_idx)}", file=sys.stderr)

    print("Reconstructing dynamic edge weights ...", file=sys.stderr)
    dyn_w = reconstruct_dynamic_edge_weights(
        base_weight=base_weight,
        edge_idx=edge_idx,
        modifier=modifier,
        default_modifier=default_modifier
    )

    print(f"Running dynamic RWR for celltype '{celltype}' ...", file=sys.stderr)
    p = random_walk_with_restart_edge_arrays(
        n_nodes=n_nodes,
        u_idx=u_idx,
        v_idx=v_idx,
        w=dyn_w,
        seed_vec=seed_vec,
        restart=args.restart,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )

    print("Mapping node scores back to regions ...", file=sys.stderr)
    out_df = map_scores_back_to_regions(
        region_ids=region_ids,
        reg_to_nodes_idx=reg_to_nodes_idx,
        reg_to_nodes_id=reg_to_nodes_id,
        p=p,
        combine_mode=args.combine_overlaps,
    )

    # carry input_score if present
    if "input_score" in regions_df.columns:
        score_map = dict(zip(regions_df["input_region"], regions_df["input_score"]))
        out_df["input_score"] = out_df["input_region"].map(score_map)

    out_df["celltype"] = celltype
    out_df["seed_mode"] = seed_mode
    out_df["floor_ct"] = floor_ct
    out_df["default_modifier"] = default_modifier

    out_df = out_df.sort_values("rwr_score", ascending=False)

    print(f"Writing output: {args.output}", file=sys.stderr)
    out_df.to_csv(args.output, sep="\t", index=False)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


