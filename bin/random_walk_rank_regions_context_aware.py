#!/usr/bin/env python3

import argparse
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyranges as pr

from enhancerinsight_report_utils_evidence import generate_dynamic_report_outputs


COORD_RE = re.compile(r'^(chr[^:]+):(\d+)-(\d+)$')
POINT_RE = re.compile(r'^(chr[^:]+):(\d+)$')


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
        ends.append(e + 1 if e == s else e)
        node_ids.append(n)
        idxs.append(i)
    return pd.DataFrame({
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
        "node_id": node_ids,
        "node_idx": idxs,
    })


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
    Load input regions while supporting both interval and SNP-style inputs.

    Coordinate-string formats (parsed per line):
      chr1:1000-2000 [score] [extra ...]
      chr1:1000      [score] [extra ...]

    Tabular formats (the WHOLE file is classified before parsing):
      interval: chr  start  end  [score] [extra ...]
      SNP:      chr  pos    [score] [extra ...]

    For tabular input, if ANY data line lacks a valid genomic end in column 3
    (column 3 is missing, non-integer, or < column 2), the entire file is
    interpreted as SNP format. SNPs use the internal region ID chr:pos-pos,
    while overlap coordinates use [pos, pos+1) so point variants can overlap
    genomic intervals correctly. ``display_region`` preserves chr:pos for
    user-facing output.
    """
    raw_rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_rows.append((line, re.split(r"\s+", line)))

    # Coordinate-string inputs are unambiguous and can be parsed per row.
    # For ordinary tabular inputs, classify the whole file as interval or SNP.
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
                end = pos + 1  # interval representation used only for overlap
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
                    end = pos + 1  # interval representation used only for overlap
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
            print(f"WARNING: cannot parse line, skipped: {line}", file=sys.stderr)
            continue

        regions.append(region)
        display_regions.append(display_region)
        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
        scores.append(score_val)

    df = pd.DataFrame({
        "input_region": regions,
        "display_region": display_regions,
        "Chromosome": chroms,
        "Start": starts,
        "End": ends,
    })
    if any_score:
        df["input_score"] = scores

    has_scores = "input_score" in df.columns and df["input_score"].notna().any()
    return df, has_scores


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


def build_seed_vector_from_region_scores(regions_df, reg_to_nodes_idx, n_nodes, combine_mode="max"):
    """Build a mass-conserving restart vector from region-level scores.

    A region's total ``input_score`` is divided equally among all unique network
    nodes overlapped by that region.  Contributions from multiple regions are
    summed at shared nodes, followed by global normalization.  ``combine_mode``
    is accepted for backward compatibility but is intentionally ignored here;
    ``--combine-overlaps`` is reserved for aggregating final node scores back to
    a region-level score.
    """
    if "input_score" not in regions_df.columns:
        return None, set()

    s = np.zeros(n_nodes, dtype=np.float64)
    scored_nodes = set()

    for row in regions_df.itertuples(index=False):
        region = getattr(row, "input_region")
        score = getattr(row, "input_score", np.nan)
        if pd.isna(score) or not np.isfinite(float(score)):
            continue
        idxs = sorted(set(int(i) for i in reg_to_nodes_idx.get(region, [])))
        if not idxs:
            continue
        share = float(score) / float(len(idxs))
        for idx in idxs:
            s[idx] += share
            if share > 0:
                scored_nodes.add(idx)

    total = float(s.sum())
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
    deg = compute_weighted_degree(n_nodes, u_idx, v_idx, w)
    p = seed_vec.astype(np.float64, copy=True)

    for it in range(max_iter):
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
            break

    p[p < 0] = 0
    s = p.sum()
    if s > 0:
        p /= s
    return p


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


def load_topology_file(path, nodes, min_component_size, use_coreness_penalty):
    topo = pd.read_csv(path, sep="\t")
    required = {"node_id", "component_size"}
    if not required.issubset(set(topo.columns)):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")

    node_to_comp = dict(zip(topo["node_id"].astype(str), topo["component_size"].astype(int)))
    comp_size_arr = np.array([node_to_comp.get(n, 1) for n in nodes], dtype=np.int32)
    valid_mask = comp_size_arr >= min_component_size

    coreness_arr = None
    coreness_penalty = None
    max_core = None

    if use_coreness_penalty:
        if "coreness" not in topo.columns:
            raise ValueError("--use-coreness-penalty was set but topology file lacks 'coreness'")
        node_to_core = dict(zip(topo["node_id"].astype(str), topo["coreness"].astype(int)))
        coreness_arr = np.array([node_to_core.get(n, 0) for n in nodes], dtype=np.int32)
        max_core = int(coreness_arr.max()) if len(coreness_arr) else 0
        if max_core <= 0:
            coreness_penalty = np.zeros(len(nodes), dtype=np.float64)
        else:
            if use_coreness_penalty == "linear":
                coreness_penalty = coreness_arr.astype(np.float64) / max_core
            elif use_coreness_penalty == "sqrt":
                coreness_penalty = np.sqrt(coreness_arr.astype(np.float64) / max_core)
            else:
                raise ValueError("Invalid coreness penalty mode")
    return valid_mask, comp_size_arr, coreness_arr, coreness_penalty, max_core


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Context-aware prioritization: dynamic-edge RWR on a cell type-specific rewired network, "
            "with component filter and optional coreness penalty. Input scores are used as initial "
            "node heat; without scores, all overlapped nodes start with equal heat."
        )
    )
    ap.add_argument("--base-edges", required=True)
    ap.add_argument("--edge-override", required=True)
    ap.add_argument("--nodes-index", required=True)
    ap.add_argument("--input-regions", required=True)

    ap.add_argument("--network", default=None,
                    help="Optional pre-built network .gpickle; only used for node annotations in the report")

    ap.add_argument("--topology-file", required=True,
                    help="node_topology.tsv.gz with node_id,component_size[,coreness]")
    ap.add_argument("--min-component-size", type=int, default=5)

    ap.add_argument(
        "--use-coreness-penalty",
        choices=["off", "linear", "sqrt"],
        default="off",
        help="Optional node-level coreness penalty applied after component filtering"
    )

    ap.add_argument("--restart", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--combine-overlaps", choices=["max", "mean", "union"], default="max")
    ap.add_argument("--output", required=True,
                    help="Output directory. The main ranked table will be written to <output>/ranking.tsv, and report-ready files to <output>/report/.")
    ap.add_argument("--report-top-n", type=int, default=20,
                    help="Number of top regions to export to report/top20.tsv. Default=20")
    ap.add_argument("--report-top-network-n", type=int, default=3,
                    help="Number of top regions for local network export. Default=3")
    ap.add_argument("--report-max-neighbors", type=int, default=25,
                    help="Maximum neighbors per overlapped node in local network export. Default=25")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_file = output_dir / "ranking.tsv"
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nodes index: {args.nodes_index}", file=sys.stderr)
    nodes, node_to_idx = load_node_index(args.nodes_index)
    n_nodes = len(nodes)

    print(f"Loading topology file: {args.topology_file}", file=sys.stderr)
    valid_mask, comp_size_arr, coreness_arr, coreness_penalty, max_core = load_topology_file(
        args.topology_file,
        nodes,
        args.min_component_size,
        args.use_coreness_penalty if args.use_coreness_penalty != "off" else None
    )
    print(
        f"Component filter: keep {valid_mask.sum()} / {len(valid_mask)} nodes "
        f"(>= {args.min_component_size})",
        file=sys.stderr
    )
    if args.use_coreness_penalty != "off":
        print(
            f"Coreness penalty: mode={args.use_coreness_penalty}, max_core={max_core}",
            file=sys.stderr
        )

    print(f"Loading base edges: {args.base_edges}", file=sys.stderr)
    u_idx, v_idx, base_weight = load_base_edges(args.base_edges)

    print(f"Loading edge override: {args.edge_override}", file=sys.stderr)
    edge_idx, modifier, default_modifier, floor_ct, celltype = load_override(args.edge_override)

    print(f"Loading input regions: {args.input_regions}", file=sys.stderr)
    regions_df, has_scores = load_input_regions(args.input_regions)
    region_ids = regions_df["input_region"].tolist()

    print("Building node coordinate table ...", file=sys.stderr)
    nodes_coord_df = build_nodes_coord_df(nodes)

    print("Building region-to-node overlap map ...", file=sys.stderr)
    reg_to_nodes_idx, reg_to_nodes_id = build_region_to_nodes_map(regions_df, nodes_coord_df)

    # The compressed arrays below remain the network used for dynamic RWR.
    # When --network is supplied, load its node attributes only for report
    # annotations such as gene_name, gene_id, gwas_snps, and
    # gwas_disease_traits.
    G = None
    node_attributes = None
    if args.network:
        print(f"Loading network node annotations: {args.network}", file=sys.stderr)
        with open(args.network, "rb") as f:
            G = pickle.load(f)
        node_attributes = {
            str(node): dict(attrs)
            for node, attrs in G.nodes(data=True)
        }

    if has_scores:
        seed_vec, _ = build_seed_vector_from_region_scores(
            regions_df, reg_to_nodes_idx, n_nodes
        )
        if seed_vec is None:
            raise ValueError("Could not build seed vector from input region scores")
        seed_mode = "input_scores"
    else:
        seed_vec, _ = build_seed_vector_from_input_regions_binary(
            reg_to_nodes_idx, region_ids, n_nodes
        )
        if seed_vec is None:
            raise ValueError("Input regions do not overlap any network nodes")
        seed_mode = "input_regions_binary"

    print(f"Seed mode = {seed_mode}", file=sys.stderr)

    print("Reconstructing dynamic edge weights ...", file=sys.stderr)
    dyn_w = reconstruct_dynamic_edge_weights(base_weight, edge_idx, modifier, default_modifier)

    print(f"Running dynamic RWR for celltype '{celltype}' ...", file=sys.stderr)
    p_raw = random_walk_with_restart_edge_arrays(
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

    # component filter
    p_component = p_raw.copy()
    p_component[~valid_mask] = 0.0
    s = p_component.sum()
    if s > 0:
        p_component /= s

    # optional coreness penalty
    if args.use_coreness_penalty != "off":
        p_adjusted = p_component * coreness_penalty
        s = p_adjusted.sum()
        if s > 0:
            p_adjusted /= s
    else:
        p_adjusted = p_component.copy()

    print("Mapping node scores back to regions ...", file=sys.stderr)
    out_raw = map_scores_back_to_regions(
        region_ids, reg_to_nodes_idx, reg_to_nodes_id, p_raw, args.combine_overlaps
    ).rename(columns={"rwr_score": "rwr_score_raw"})

    out_component = map_scores_back_to_regions(
        region_ids, reg_to_nodes_idx, reg_to_nodes_id, p_component, args.combine_overlaps
    ).rename(columns={"rwr_score": "rwr_score_component_filtered"})

    out_adjusted = map_scores_back_to_regions(
        region_ids, reg_to_nodes_idx, reg_to_nodes_id, p_adjusted, args.combine_overlaps
    ).rename(columns={"rwr_score": "rwr_score_adjusted"})

    out_df = out_raw.merge(
        out_component[["input_region", "rwr_score_component_filtered"]],
        on="input_region",
        how="left"
    ).merge(
        out_adjusted[["input_region", "rwr_score_adjusted"]],
        on="input_region",
        how="left"
    )

    keep_cols = ["input_region", "n_overlapped_nodes", "overlapped_nodes", "overlapped_node_scores"]
    for c in keep_cols:
        if c in out_raw.columns and c not in out_df.columns:
            out_df[c] = out_raw[c]

    if "input_score" in regions_df.columns:
        score_map = dict(zip(regions_df["input_region"], regions_df["input_score"]))
        out_df["input_score"] = out_df["input_region"].map(score_map)

    out_df["celltype"] = celltype
    out_df["seed_mode"] = seed_mode
    out_df["floor_ct"] = floor_ct
    out_df["default_modifier"] = default_modifier
    out_df["min_component_size"] = args.min_component_size
    out_df["coreness_penalty_mode"] = args.use_coreness_penalty
    out_df["topology_component_kept"] = out_df["input_region"].map(
        lambda r: int(any(comp_size_arr[idx] >= args.min_component_size for idx in reg_to_nodes_idx.get(r, [])))
    )

    # Remove regions with no network overlap before any ranking is derived.
    out_df = out_df[pd.to_numeric(out_df["n_overlapped_nodes"], errors="coerce").fillna(0) > 0].copy()
    if "input_score" in out_df.columns:
        out_df["_input_score_sort"] = pd.to_numeric(
            out_df["input_score"], errors="coerce"
        ).fillna(-np.inf)
    else:
        out_df["_input_score_sort"] = -np.inf
    out_df = out_df.sort_values(
        ["rwr_score_adjusted", "_input_score_sort"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns="_input_score_sort").reset_index(drop=True)

    # Restore chr:pos display for SNP inputs only after all internal mapping,
    # scoring, filtering, and ranking have finished.
    display_map = dict(zip(regions_df["input_region"], regions_df["display_region"]))
    out_df["input_region"] = out_df["input_region"].map(lambda r: display_map.get(r, r))

    print(f"Writing output: {ranking_file}", file=sys.stderr)
    out_df.to_csv(ranking_file, sep="\t", index=False)

    print(f"Writing report-ready tables to {report_dir}", file=sys.stderr)
    generate_dynamic_report_outputs(
        out_df=out_df,
        output_tsv=str(ranking_file),
        args=args,
        nodes=nodes,
        node_to_idx=node_to_idx,
        u_idx=u_idx,
        v_idx=v_idx,
        base_weight=base_weight,
        dyn_weight=dyn_w,
        edge_idx=edge_idx,
        modifier=modifier,
        p_adjusted=p_adjusted,
        seed_vector=seed_vec,
        node_attributes=node_attributes,
        celltype=celltype,
        seed_mode=seed_mode,
        report_dir=str(report_dir),
        top_n=args.report_top_n,
        top_network_n=args.report_top_network_n,
        max_neighbors_per_node=args.report_max_neighbors,
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


