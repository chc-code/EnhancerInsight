#!/usr/bin/env python3
"""
EnhancerInsight prioritization report utilities.

This module generates the final Report API v2 files from already-computed
prioritization outputs. It does not change any ranking algorithm or recompute
prioritization scores.

Expected report layout:

report/
  metadata.json
  summary.tsv
  ranking.tsv
  top20.tsv
  evidence_summary.tsv
  evidence_details.tsv
  candidate_cards.tsv
  algorithm_trace.tsv
  rank_change.tsv
  network_summary.tsv
  network/
    region001_nodes.tsv
    region001_edges.tsv
    ...
  figures/          # created empty; figures are generated later by Rmd
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

COORD_RE = re.compile(r"^(chr[^:]+):(\d+)-(\d+)$")


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    return str(x) == ""


def _format_value(x: Any, digits: int = 6) -> str:
    if _is_missing(x):
        return ""
    try:
        xf = float(x)
        if math.isnan(xf):
            return ""
        return f"{xf:.{digits}g}"
    except Exception:
        return str(x)


def _split_semicolon(x: Any) -> List[str]:
    if _is_missing(x):
        return []
    return [s.strip() for s in str(x).split(";") if s.strip()]


def _parse_numeric_list(x: Any) -> List[float]:
    vals: List[float] = []
    for item in _split_semicolon(x):
        try:
            vals.append(float(item))
        except Exception:
            pass
    return vals


def _rank_desc(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-np.inf)
    return numeric.rank(method="min", ascending=False).astype(int)


def _sanitize_filename(s: Any, max_len: int = 80) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    clean = clean.strip("_") or "region"
    return clean[:max_len]


def infer_node_type(node_id: str, attrs: Optional[Mapping[str, Any]] = None) -> str:
    attrs = attrs or {}
    for key in ["node_type", "type", "category", "node_class", "biotype"]:
        val = attrs.get(key)
        if not _is_missing(val):
            return str(val)
    nid = str(node_id)
    if COORD_RE.match(nid):
        return "region"
    if nid.lower().startswith("rs"):
        return "variant"
    if nid.lower().startswith("ensg") or attrs.get("gene_name") or attrs.get("symbol"):
        return "gene"
    return "node"



def _node_display_label(node_id: Any, G: Any = None) -> str:
    """Return a readable label for a network node when attributes are available."""
    nid = str(node_id)
    attrs: Mapping[str, Any] = {}
    try:
        if G is not None and nid in G:
            attrs = dict(G.nodes[nid])
        elif G is not None and node_id in G:
            attrs = dict(G.nodes[node_id])
    except Exception:
        attrs = {}

    for key in [
        "gene_symbol", "symbol", "gene_name", "external_gene_name",
        "name", "label", "display_name",
    ]:
        val = attrs.get(key)
        if not _is_missing(val):
            return str(val)
    return nid


def _highest_scoring_overlapped_node(
    row: Mapping[str, Any],
    node_score_map: Optional[Mapping[str, float]] = None,
    G: Any = None,
) -> Optional[Dict[str, Any]]:
    """Find the overlapped node with the highest available node score.

    Priority of score sources:
    1. row["overlapped_node_scores"] when it is aligned to row["overlapped_nodes"]
    2. node_score_map, which is the same score source used for local_network_nodes.tsv

    If no score is available, return the first overlapped node with an empty score,
    but mark the source as "first_node_no_score" so downstream text does not call it
    highest-scoring by mistake.
    """
    nodes = _split_semicolon(row.get("overlapped_nodes", ""))
    if not nodes:
        return None

    scores_from_row = _parse_numeric_list(row.get("overlapped_node_scores", ""))
    scored: List[Tuple[str, float, str]] = []

    if len(scores_from_row) == len(nodes):
        scored = [(str(n), float(s), "overlapped_node_scores") for n, s in zip(nodes, scores_from_row)]
    elif node_score_map is not None:
        for n in nodes:
            key = str(n)
            if key in node_score_map and not _is_missing(node_score_map[key]):
                scored.append((key, _safe_float(node_score_map[key], np.nan), "node_score_map"))

    if scored:
        scored = [(n, s, src) for n, s, src in scored if not math.isnan(float(s))]

    if scored:
        node_id, score, source = max(scored, key=lambda x: x[1])
        return {
            "node_id": node_id,
            "label": _node_display_label(node_id, G),
            "score": score,
            "score_source": source,
        }

    node_id = str(nodes[0])
    return {
        "node_id": node_id,
        "label": _node_display_label(node_id, G),
        "score": "",
        "score_source": "first_node_no_score",
    }


def _format_node_with_score(node_info: Optional[Mapping[str, Any]]) -> str:
    if not node_info:
        return ""
    node_id = str(node_info.get("node_id", ""))
    label = str(node_info.get("label", node_id))
    score = node_info.get("score", "")
    if label and label != node_id:
        node_text = f"{label} ({node_id})"
    else:
        node_text = node_id
    if not _is_missing(score):
        node_text += f"; score={_format_value(score)}"
    return node_text

def _edge_weight_from_attrs(attrs: Mapping[str, Any]) -> float:
    for key in ["weight", "score", "confidence", "base_weight"]:
        if key in attrs:
            return _safe_float(attrs.get(key), 1.0)
    return 1.0


def _edge_type_from_attrs(attrs: Mapping[str, Any]) -> str:
    for key in ["edge_type", "type", "relation", "source", "evidence"]:
        val = attrs.get(key)
        if not _is_missing(val):
            return str(val)
    return "network_edge"


def _dense_rank_lexicographic(final_scores: pd.Series, input_scores: pd.Series) -> pd.Series:
    """Dense-rank candidates by (final score desc, input score desc).

    Candidates are tied only when both values are equal (including paired NaNs
    after normalization to a common sentinel).
    """
    final_num = pd.to_numeric(final_scores, errors="coerce")
    input_num = pd.to_numeric(input_scores, errors="coerce")
    keys = list(zip(final_num.fillna(-np.inf), input_num.fillna(-np.inf)))
    unique_keys = sorted(set(keys), key=lambda x: (x[0], x[1]), reverse=True)
    rank_map = {key: i + 1 for i, key in enumerate(unique_keys)}
    return pd.Series([rank_map[key] for key in keys], index=final_scores.index, dtype=int)


def _filter_overlapped_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only input regions mapped to at least one network node."""
    if "n_overlapped_nodes" in df.columns:
        mask = pd.to_numeric(df["n_overlapped_nodes"], errors="coerce").fillna(0) > 0
        return df.loc[mask].copy()
    if "overlapped_nodes" in df.columns:
        mask = df["overlapped_nodes"].fillna("").astype(str).str.strip().ne("")
        return df.loc[mask].copy()
    return df.copy()


def _filtered_input_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Return input regions excluded because they do not map to a network node."""
    if "n_overlapped_nodes" in df.columns:
        mask = pd.to_numeric(df["n_overlapped_nodes"], errors="coerce").fillna(0) <= 0
        return df.loc[mask].copy()
    if "overlapped_nodes" in df.columns:
        mask = df["overlapped_nodes"].fillna("").astype(str).str.strip().eq("")
        return df.loc[mask].copy()
    return df.iloc[0:0].copy()


def _write_filtered_inputs(report_dir: Path, out_df: pd.DataFrame) -> int:
    """Write inputs excluded from ranking and return their number."""
    filtered = _filtered_input_regions(out_df)
    filtered.to_csv(report_dir / "filtered_inputs.tsv", sep="\t", index=False)
    return int(len(filtered))


def _prepare_ranking(out_df: pd.DataFrame, score_col: str, top_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if score_col not in out_df.columns:
        raise ValueError(f"Expected score column '{score_col}' in ranking output")

    # Filtering precedes all ranks so input ranks remain continuous in reports.
    ranking = _filter_overlapped_regions(out_df)
    input_scores = ranking["input_score"] if "input_score" in ranking.columns else pd.Series(np.nan, index=ranking.index)
    ranking["_input_score_sort"] = pd.to_numeric(input_scores, errors="coerce").fillna(-np.inf)
    ranking = ranking.sort_values(
        [score_col, "_input_score_sort"],
        ascending=[False, False],
        kind="mergesort",
    ).drop(columns="_input_score_sort").reset_index(drop=True)

    if "rank" in ranking.columns:
        ranking = ranking.drop(columns=["rank"])
    ranking.insert(
        0,
        "rank",
        _dense_rank_lexicographic(
            ranking[score_col],
            ranking["input_score"] if "input_score" in ranking.columns else pd.Series(np.nan, index=ranking.index),
        ).to_numpy(),
    )
    return ranking, ranking.head(top_n).copy()


def _write_metadata(report_dir: Path, metadata: Mapping[str, Any]) -> None:
    with open(report_dir / "metadata.json", "w") as f:
        json.dump(dict(metadata), f, indent=2, default=str)


def _write_empty_tsv(path: Path, columns: Sequence[str]) -> None:
    pd.DataFrame(columns=list(columns)).to_csv(path, sep="\t", index=False)


# -----------------------------------------------------------------------------
# Core report tables
# -----------------------------------------------------------------------------

def _write_summary(report_dir: Path, rows: Sequence[Tuple[str, Any]]) -> None:
    pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(
        report_dir / "summary.tsv", sep="\t", index=False
    )


def _score_col_for_mode(mode: str, out_df: pd.DataFrame) -> str:
    if "rwr_score_adjusted" in out_df.columns:
        return "rwr_score_adjusted"
    if "rwr_score" in out_df.columns:
        return "rwr_score"
    if "final_score" in out_df.columns:
        return "final_score"
    raise ValueError("Cannot determine final score column from output table")


def _interpret_candidate(row: Mapping[str, Any], score_col: str, mode: str) -> str:
    n_nodes = _safe_float(row.get("n_overlapped_nodes", 0), 0)
    final_score = row.get(score_col, "")
    parts: List[str] = []
    if n_nodes > 1:
        parts.append("Supported by multiple overlapping network nodes")
    elif n_nodes == 1:
        parts.append("Supported by one overlapping network node")
    else:
        parts.append("No direct overlapping network node was detected")
    if "rwr_score_adjusted" in row:
        parts.append("after dynamic network and topology adjustment")
    elif "input_score" in row and not _is_missing(row.get("input_score")):
        parts.append("after network propagation from input scores")
    else:
        parts.append("after network propagation")
    if not _is_missing(final_score):
        parts.append(f"with final score {_format_value(final_score)}")
    return "; ".join(parts) + "."


def _write_evidence_summary(report_dir: Path, top: pd.DataFrame, score_col: str, mode: str, node_score_map: Optional[Mapping[str, float]] = None, G: Any = None) -> None:
    rows = []
    for _, row in top.iterrows():
        top_node = _highest_scoring_overlapped_node(row, node_score_map=node_score_map, G=G)
        rows.append({
            "Region": row.get("input_region", ""),
            "FinalScore": row.get(score_col, ""),
            "HighestScoringOverlappedNode": _format_node_with_score(top_node),
            "Interpretation": _interpret_candidate(row, score_col, mode),
        })
    pd.DataFrame(rows, columns=["Region", "FinalScore", "HighestScoringOverlappedNode", "Interpretation"]).to_csv(
        report_dir / "evidence_summary.tsv", sep="\t", index=False
    )


def _write_evidence_details(report_dir: Path, top: pd.DataFrame, score_col: str, mode: str, node_score_map: Optional[Mapping[str, float]] = None, G: Any = None) -> None:
    metric_candidates = [
        ("Final rank", "rank"),
        ("Final score", score_col),
        ("Input score", "input_score"),
        ("Overlap nodes", "n_overlapped_nodes"),
        ("Overlapped nodes", "overlapped_nodes"),
        ("Overlapped node scores", "overlapped_node_scores"),
        ("Raw RWR", "rwr_score_raw"),
        ("Component-filtered RWR", "rwr_score_component_filtered"),
        ("Adjusted RWR", "rwr_score_adjusted"),
        ("Component sizes", "component_sizes"),
        ("Minimum component size", "min_component_size"),
        ("Maximum component size", "max_component_size"),
        ("Any tiny component", "any_tiny_component"),
        ("All tiny components", "all_tiny_components"),
        ("Topology component kept", "topology_component_kept"),
        ("Cell type", "celltype"),
        ("Seed mode", "seed_mode"),
        ("Cell-type floor", "floor_ct"),
        ("Default modifier", "default_modifier"),
        ("Coreness penalty mode", "coreness_penalty_mode"),
    ]
    rows = []
    for _, row in top.iterrows():
        region = row.get("input_region", "")
        for metric, col in metric_candidates:
            if col in top.columns:
                value = row.get(col, "")
                if not _is_missing(value):
                    rows.append({"Region": region, "Metric": metric, "Value": value})
        # Derived descriptive metrics from existing values only.
        top_node = _highest_scoring_overlapped_node(row, node_score_map=node_score_map, G=G)
        if top_node and top_node.get("score_source") != "first_node_no_score":
            rows.append({"Region": region, "Metric": "Highest-scoring overlapped node", "Value": _format_node_with_score(top_node)})
            rows.append({"Region": region, "Metric": "Highest overlapped node score", "Value": top_node.get("score", "")})
        elif top_node:
            rows.append({"Region": region, "Metric": "First overlapped node", "Value": _format_node_with_score(top_node)})

        scores = _parse_numeric_list(row.get("overlapped_node_scores", ""))
        if scores:
            rows.append({"Region": region, "Metric": "Mean overlap score", "Value": float(np.mean(scores))})
            rows.append({"Region": region, "Metric": "Maximum overlap score", "Value": float(np.max(scores))})
    pd.DataFrame(rows, columns=["Region", "Metric", "Value"]).to_csv(
        report_dir / "evidence_details.tsv", sep="\t", index=False
    )


def _write_candidate_cards(report_dir: Path, top: pd.DataFrame, score_col: str, mode: str, node_score_map: Optional[Mapping[str, float]] = None, G: Any = None) -> None:
    rows = []
    for _, row in top.iterrows():
        region = row.get("input_region", "")
        bullets = [f"Region: {region}", f"Final score: {_format_value(row.get(score_col, ''))}"]
        if "rank" in row:
            bullets.insert(0, f"Rank: {row.get('rank')}")
        if "n_overlapped_nodes" in row:
            bullets.append(f"Overlaps {row.get('n_overlapped_nodes')} network node(s)")
        top_node = _highest_scoring_overlapped_node(row, node_score_map=node_score_map, G=G)
        if top_node and top_node.get("score_source") != "first_node_no_score":
            bullets.append(f"Highest-scoring overlapped node: {_format_node_with_score(top_node)}")
        elif top_node:
            bullets.append(f"First overlapped node: {_format_node_with_score(top_node)}")
        if "input_score" in row and not _is_missing(row.get("input_score")):
            bullets.append(f"Input score: {_format_value(row.get('input_score'))}")
        if "rwr_score_raw" in row:
            bullets.append(f"Raw RWR: {_format_value(row.get('rwr_score_raw'))}")
        if "rwr_score_component_filtered" in row:
            bullets.append(f"Component-filtered RWR: {_format_value(row.get('rwr_score_component_filtered'))}")
        if "topology_component_kept" in row:
            bullets.append(f"Topology component kept: {row.get('topology_component_kept')}")
        if "celltype" in row:
            bullets.append(f"Cell type: {row.get('celltype')}")
        bullets.append(_interpret_candidate(row, score_col, mode))
        rows.append({"Region": region, "Card": " | ".join([str(x) for x in bullets if str(x).strip()])})
    pd.DataFrame(rows, columns=["Region", "Card"]).to_csv(
        report_dir / "candidate_cards.tsv", sep="\t", index=False
    )


def _write_algorithm_trace_static(report_dir: Path, top: pd.DataFrame, score_col: str) -> None:
    rows = []
    for _, row in top.iterrows():
        region = row.get("input_region", "")
        if "input_score" in top.columns and not _is_missing(row.get("input_score")):
            rows.append({"Region": region, "Step": "Input score", "Value": row.get("input_score")})
        if "n_overlapped_nodes" in top.columns:
            rows.append({"Region": region, "Step": "Overlap mapping", "Value": row.get("n_overlapped_nodes")})
        rows.append({"Region": region, "Step": "RWR score", "Value": row.get(score_col, "")})
        rows.append({"Region": region, "Step": "Final score", "Value": row.get(score_col, "")})
    pd.DataFrame(rows, columns=["Region", "Step", "Value"]).to_csv(
        report_dir / "algorithm_trace.tsv", sep="\t", index=False
    )


def _write_algorithm_trace_dynamic(report_dir: Path, top: pd.DataFrame, score_col: str) -> None:
    rows = []
    for _, row in top.iterrows():
        region = row.get("input_region", "")
        if "input_score" in top.columns and not _is_missing(row.get("input_score")):
            rows.append({"Region": region, "Step": "Input score", "Value": row.get("input_score")})
        if "rwr_score_raw" in top.columns:
            rows.append({"Region": region, "Step": "Raw RWR", "Value": row.get("rwr_score_raw")})
        if "rwr_score_component_filtered" in top.columns:
            rows.append({"Region": region, "Step": "Component filtering", "Value": row.get("rwr_score_component_filtered")})
        if "rwr_score_component_filtered" in top.columns and "rwr_score_adjusted" in top.columns:
            comp = _safe_float(row.get("rwr_score_component_filtered"), np.nan)
            adj = _safe_float(row.get("rwr_score_adjusted"), np.nan)
            ratio = adj / comp if comp and not math.isnan(comp) else ""
            rows.append({"Region": region, "Step": "Topology adjustment", "Value": ratio})
        rows.append({"Region": region, "Step": "Final score", "Value": row.get(score_col, "")})
    pd.DataFrame(rows, columns=["Region", "Step", "Value"]).to_csv(
        report_dir / "algorithm_trace.tsv", sep="\t", index=False
    )


def _write_rank_change_static(report_dir: Path, ranking: pd.DataFrame, score_col: str) -> None:
    cols = ["Region", "InputRank", "FinalRank", "RankChange", "InputScore", "FinalScore"]
    if "input_score" not in ranking.columns:
        _write_empty_tsv(report_dir / "rank_change.tsv", cols)
        return
    tmp = ranking.copy()
    numeric = pd.to_numeric(tmp["input_score"], errors="coerce")
    if numeric.notna().sum() == 0:
        _write_empty_tsv(report_dir / "rank_change.tsv", cols)
        return
    tmp["InputRank"] = _rank_desc(numeric)
    tmp["FinalRank"] = tmp["rank"]
    tmp["RankChange"] = tmp["InputRank"] - tmp["FinalRank"]
    out = pd.DataFrame({
        "Region": tmp["input_region"],
        "InputRank": tmp["InputRank"],
        "FinalRank": tmp["FinalRank"],
        "RankChange": tmp["RankChange"],
        "InputScore": tmp["input_score"],
        "FinalScore": tmp[score_col],
    })
    out.to_csv(report_dir / "rank_change.tsv", sep="\t", index=False)


def _write_rank_change_dynamic(report_dir: Path, ranking: pd.DataFrame, score_col: str) -> None:
    cols = ["Region", "RawRank", "AdjustedRank", "RankChange", "RawScore", "AdjustedScore"]
    if "rwr_score_raw" not in ranking.columns or score_col not in ranking.columns:
        _write_empty_tsv(report_dir / "rank_change.tsv", cols)
        return
    tmp = ranking.copy()
    tmp["RawRank"] = _rank_desc(tmp["rwr_score_raw"])
    tmp["AdjustedRank"] = tmp["rank"]
    tmp["RankChange"] = tmp["RawRank"] - tmp["AdjustedRank"]
    out = pd.DataFrame({
        "Region": tmp["input_region"],
        "RawRank": tmp["RawRank"],
        "AdjustedRank": tmp["AdjustedRank"],
        "RankChange": tmp["RankChange"],
        "RawScore": tmp["rwr_score_raw"],
        "AdjustedScore": tmp[score_col],
    })
    out.to_csv(report_dir / "rank_change.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Local network export
# -----------------------------------------------------------------------------

def _static_edge_attrs(G: Any, u: Any, v: Any) -> Mapping[str, Any]:
    attrs = G.get_edge_data(u, v, default={}) or {}
    # MultiGraph compatibility: {key: attrdict}
    if isinstance(attrs, dict) and attrs and all(isinstance(val, dict) for val in attrs.values()):
        return list(attrs.values())[0]
    return attrs


def _blend_hex(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    a = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5)); b = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
    rgb = tuple(round(x + (y-x)*t) for x, y in zip(a,b))
    return "#%02X%02X%02X" % rgb


def _seed_color_map(seed_vector: Sequence[float]) -> Dict[int, str]:
    sv = np.asarray(seed_vector, dtype=float); idx = np.where(np.isfinite(sv) & (sv > 0))[0]
    if len(idx) == 0: return {}
    vals = sv[idx]; lo, hi = float(vals.min()), float(vals.max())
    scaled = np.ones(len(vals))*0.5 if hi <= lo else (vals-lo)/(hi-lo)
    return {int(i): _blend_hex("#C6DBEF", "#08519C", 0.15 + 0.85*float(s)) for i,s in zip(idx,scaled)}


def _raw_seed_heat_from_report_inputs(
    inputs: pd.DataFrame,
    node_to_idx: Mapping[str, int],
    n_nodes: int,
    excluded_focal_idxs: Optional[Iterable[int]] = None,
    fallback_seed_vector: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Rebuild raw input heat exactly as the standalone network exporters do.

    Each scored input region contributes its *raw* input score, split equally
    among all unique overlapped network nodes. Contributions from different
    input regions are summed.  For a focal candidate, any input region that
    overlaps one of the focal nodes can be excluded from the non-focal seed set.

    This function intentionally does NOT use the normalized RWR restart vector
    for visualization seed discovery.  The standalone exporters reconstruct
    seeds from the original region-to-node mapping, and the report network must
    follow the same rule.
    """
    heat = np.zeros(int(n_nodes), dtype=float)
    focal = set(int(x) for x in (excluded_focal_idxs or []))

    has_scores = (
        isinstance(inputs, pd.DataFrame)
        and "input_score" in inputs.columns
        and pd.to_numeric(inputs["input_score"], errors="coerce").notna().any()
        and "overlapped_nodes" in inputs.columns
    )

    if has_scores:
        for _, row in inputs.iterrows():
            score = _safe_float(row.get("input_score"), np.nan)
            if not np.isfinite(score) or score <= 0:
                continue
            idxs = sorted({
                int(node_to_idx[str(n)])
                for n in _split_semicolon(row.get("overlapped_nodes", ""))
                if str(n) in node_to_idx
            })
            if not idxs:
                continue
            # Match standalone behavior: if this input region maps to ANY focal
            # node, exclude the whole input region from the non-focal seed heat.
            if focal and focal.intersection(idxs):
                continue
            share = float(score) / float(len(idxs))
            for idx in idxs:
                heat[idx] += share
        return heat

    # Backward-compatible fallback for seed modes without input scores.
    if fallback_seed_vector is not None:
        arr = np.asarray(fallback_seed_vector, dtype=float).reshape(-1)
        if len(arr) == int(n_nodes):
            arr = arr.copy()
            arr[~np.isfinite(arr)] = 0.0
            arr[arr < 0] = 0.0
            if focal:
                arr[list(focal)] = 0.0
            return arr
    return heat


def _build_path_cost_graph(
    n_nodes: int,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    edge_weight: np.ndarray,
    metric: str = "inverse_weight",
) -> csr_matrix:
    """Sparse undirected path-cost graph matching the standalone exporters."""
    uu = np.asarray(u_idx, dtype=int)
    vv = np.asarray(v_idx, dtype=int)
    ww = np.asarray(edge_weight, dtype=float)
    positive = np.isfinite(ww) & (ww > 0)
    uu, vv, ww = uu[positive], vv[positive], ww[positive]
    if metric == "inverse_weight":
        cost = 1.0 / np.maximum(ww, 1e-12)
    elif metric == "hops":
        cost = np.ones_like(ww, dtype=float)
    else:
        raise ValueError(f"Unknown path metric: {metric}")
    rows = np.concatenate([uu, vv])
    cols = np.concatenate([vv, uu])
    data = np.concatenate([cost, cost])
    return csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))


def _reconstruct_standalone_path(target_idx: int, predecessors: np.ndarray, focal_set: set[int]) -> Optional[List[int]]:
    path = [int(target_idx)]
    cur = int(target_idx)
    seen = {cur}
    while cur not in focal_set:
        prev = int(predecessors[cur])
        if prev == -9999:
            return None
        if prev in seen:
            return None
        path.append(prev)
        seen.add(prev)
        cur = prev
    path.reverse()
    return path


def _extract_standalone_paths(
    n_nodes: int,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    edge_weight: np.ndarray,
    focal_idxs: Sequence[int],
    seed_idxs: Sequence[int],
    max_hops: Optional[int] = 2,
    metric: str = "inverse_weight",
) -> Dict[int, Dict[str, Any]]:
    """Exact standalone logic: weighted best path first, then max-hop filter."""
    if not focal_idxs or not seed_idxs:
        return {}
    cost_graph = _build_path_cost_graph(n_nodes, u_idx, v_idx, edge_weight, metric)
    dist, predecessors, sources = dijkstra(
        cost_graph,
        directed=False,
        indices=np.asarray(list(focal_idxs), dtype=np.int32),
        return_predecessors=True,
        min_only=True,
    )
    focal_set = set(int(x) for x in focal_idxs)
    paths: Dict[int, Dict[str, Any]] = {}
    for seed_idx in seed_idxs:
        seed_idx = int(seed_idx)
        if not np.isfinite(dist[seed_idx]) or int(sources[seed_idx]) == -9999:
            continue
        path = _reconstruct_standalone_path(seed_idx, predecessors, focal_set)
        if path is None:
            continue
        if max_hops is not None and (len(path) - 1) > int(max_hops):
            continue
        paths[seed_idx] = {
            "path": path,
            "distance": float(dist[seed_idx]),
            "source_focal_idx": int(sources[seed_idx]),
        }
    return paths


def _path_nodes_and_pairs(paths: Mapping[int, Mapping[str, Any]]) -> Tuple[set[int], set[Tuple[int, int]]]:
    nodes: set[int] = set()
    pairs: set[Tuple[int, int]] = set()
    for info in paths.values():
        p = [int(x) for x in info.get("path", [])]
        nodes.update(p)
        for a, b in zip(p[:-1], p[1:]):
            pairs.add((min(a, b), max(a, b)))
    return nodes, pairs


def _edge_records_for_path_pairs(
    path_pairs: set[Tuple[int, int]],
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    primary_weight: np.ndarray,
    base_weight: Optional[np.ndarray] = None,
    modifier_by_edge: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Keep the strongest compressed edge for every selected node pair."""
    if not path_pairs:
        return []
    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pw = np.asarray(primary_weight, dtype=float)
    bw = np.asarray(base_weight, dtype=float) if base_weight is not None else None
    mod = np.asarray(modifier_by_edge, dtype=float) if modifier_by_edge is not None else None
    for ei, (u, v) in enumerate(zip(np.asarray(u_idx, dtype=int), np.asarray(v_idx, dtype=int))):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in path_pairs:
            continue
        rec = {
            "edge_index": int(ei),
            "u_idx": int(u),
            "v_idx": int(v),
            "weight": float(pw[ei]),
        }
        if bw is not None:
            rec["base_weight"] = float(bw[ei])
            rec["dynamic_weight"] = float(pw[ei])
        if mod is not None and ei < len(mod):
            rec["modifier"] = float(mod[ei]) if np.isfinite(mod[ei]) else ""
        if key not in best or rec["weight"] > best[key]["weight"]:
            best[key] = rec
    return list(best.values())


def _seed_color_from_raw_heat(displayed_idxs: Sequence[int], heat_values: np.ndarray) -> Dict[int, str]:
    idxs = [int(i) for i in displayed_idxs]
    if not idxs:
        return {}
    vals = np.asarray([float(heat_values[i]) for i in idxs], dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    scaled = np.full(len(vals), 0.5, dtype=float) if hi <= lo else (vals - lo) / (hi - lo)
    return {
        idx: _blend_hex("#C6DBEF", "#08519C", 0.15 + 0.85 * float(z))
        for idx, z in zip(idxs, scaled)
    }


def _export_static_networks(
    report_dir, top, G, node_score_map, top_network_n, max_neighbors_per_node,
    seed_vector=None, node_list=None, node_attributes=None, max_hops=2,
    report_inputs: Optional[pd.DataFrame] = None,
):
    """Export Mode2A focal-to-seed subnetworks using standalone logic."""
    network_dir = _mkdir(report_dir / "network")
    rows_summary: List[Dict[str, Any]] = []
    summary_cols = ["Rank", "Region", "NodesFile", "EdgesFile", "NumberNodes", "NumberEdges"]
    node_cols = ["node_id", "label", "gene_id", "gene_name", "role", "initial_heat", "node_color", "node_size", "node_shape", "path_distance", "path_hops", "gwas_snps", "gwas_disease_traits"]
    edge_cols = ["source", "target", "weight", "edge_type"]

    if G is None or top_network_n <= 0:
        _write_empty_tsv(report_dir / "network_summary.tsv", summary_cols)
        return

    node_list = [str(n) for n in (node_list or list(G.nodes()))]
    node_to_idx = {str(n): i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)
    su, sv, sw = _static_edge_arrays(G, node_list, node_to_idx)
    inputs = report_inputs if isinstance(report_inputs, pd.DataFrame) else top

    # All-input heat is used only to display focal heat when focal is itself an input seed.
    heat_with_focal = _raw_seed_heat_from_report_inputs(
        inputs, node_to_idx, n_nodes, excluded_focal_idxs=None, fallback_seed_vector=seed_vector
    )

    for rank_idx, (_, row) in enumerate(top.head(top_network_n).iterrows(), start=1):
        region = row.get("input_region", "")
        focal_idxs = sorted({
            int(node_to_idx[n])
            for n in _split_semicolon(row.get("overlapped_nodes", ""))
            if n in node_to_idx
        })
        if not focal_idxs:
            focal_id = str(row.get("max_rwr_node", "") or "")
            if focal_id in node_to_idx:
                focal_idxs = [int(node_to_idx[focal_id])]
        focal_set = set(focal_idxs)

        # Standalone behavior: exclude the entire input region from non-focal seed
        # heat when any mapped node of that region overlaps a focal node.
        seed_heat_raw = _raw_seed_heat_from_report_inputs(
            inputs, node_to_idx, n_nodes, excluded_focal_idxs=focal_set,
            fallback_seed_vector=seed_vector,
        )
        seed_idxs = [int(i) for i in np.flatnonzero(seed_heat_raw > 0) if int(i) not in focal_set]

        paths = _extract_standalone_paths(
            n_nodes=n_nodes, u_idx=su, v_idx=sv, edge_weight=sw,
            focal_idxs=focal_idxs, seed_idxs=seed_idxs,
            max_hops=max_hops, metric="inverse_weight",
        ) if focal_idxs and seed_idxs else {}

        reachable_seed_idxs = sorted(paths.keys())
        path_node_set, path_pairs = _path_nodes_and_pairs(paths)
        path_node_set.update(focal_idxs)
        path_node_set.update(reachable_seed_idxs)
        edge_records_raw = _edge_records_for_path_pairs(path_pairs, su, sv, sw)

        displayed_seed_idxs = sorted(set(reachable_seed_idxs) | {
            i for i in focal_idxs if heat_with_focal[i] > 0
        })
        color_map = _seed_color_from_raw_heat(displayed_seed_idxs, heat_with_focal)

        # Store shortest selected path metadata for reachable seeds.
        path_meta = {
            int(si): (float(info["distance"]), len(info["path"]) - 1)
            for si, info in paths.items()
        }

        node_records: List[Dict[str, Any]] = []
        for idx in sorted(path_node_set):
            nid = node_list[idx]
            attrs = dict(G.nodes[nid]) if nid in G else dict((node_attributes or {}).get(nid, {}) or {})
            gene_name = attrs.get("gene_name", "")
            gene_id = attrs.get("gene_id", "")
            # Match the validated standalone exporter: classify a node as a gene
            # when gene_id is present, with ENSG node IDs as a fallback.
            is_gene = (
                not _is_missing(gene_id)
                or str(nid).upper().startswith("ENSG")
            )
            label = gene_name if not _is_missing(gene_name) else nid
            is_focal = idx in focal_set
            is_seed = idx in paths
            is_seed_visual = is_seed or (is_focal and heat_with_focal[idx] > 0)
            role = "focal" if is_focal else ("seed" if is_seed else "intermediate")
            heat = float(heat_with_focal[idx]) if is_seed_visual else ""
            dist_val, hops_val = path_meta.get(idx, ("", ""))
            node_records.append({
                "node_id": nid,
                "label": label,
                "gene_id": gene_id,
                "gene_name": gene_name,
                "role": role,
                "initial_heat": heat,
                "node_color": color_map.get(idx, "#D9D9D9") if is_seed_visual else "#D9D9D9",
                "node_size": 42 if (is_focal or is_seed) else 26,
                "node_shape": "ellipse" if is_gene else "triangle",
                "path_distance": dist_val,
                "path_hops": hops_val,
                "gwas_snps": attrs.get("gwas_snps", ""),
                "gwas_disease_traits": attrs.get("gwas_disease_traits", ""),
            })

        edge_records: List[Dict[str, Any]] = []
        for rec in edge_records_raw:
            u_id, v_id = node_list[rec["u_idx"]], node_list[rec["v_idx"]]
            attrs = _static_edge_attrs(G, u_id, v_id)
            edge_records.append({
                "source": u_id,
                "target": v_id,
                "weight": float(rec["weight"]),
                "edge_type": _edge_type_from_attrs(attrs),
            })

        prefix = f"region{rank_idx:03d}"
        nf = network_dir / f"{prefix}_nodes.tsv"
        ef = network_dir / f"{prefix}_edges.tsv"
        pd.DataFrame(node_records, columns=node_cols).to_csv(nf, sep="\t", index=False)
        pd.DataFrame(edge_records, columns=edge_cols).to_csv(ef, sep="\t", index=False)
        rows_summary.append({
            "Rank": rank_idx,
            "Region": region,
            "NodesFile": f"network/{nf.name}",
            "EdgesFile": f"network/{ef.name}",
            "NumberNodes": len(node_records),
            "NumberEdges": len(edge_records),
        })

    pd.DataFrame(rows_summary, columns=summary_cols).to_csv(report_dir / "network_summary.tsv", sep="\t", index=False)


def _export_dynamic_networks(
    report_dir, top, nodes, node_to_idx, u_idx, v_idx, base_weight, dyn_weight, modifier_by_edge,
    p_adjusted, top_network_n, max_neighbors_per_node, seed_vector=None, node_attributes=None,
    max_hops=2, report_inputs: Optional[pd.DataFrame] = None,
):
    """Export Mode2B focal-to-seed subnetworks using the same standalone logic."""
    network_dir = _mkdir(report_dir / "network")
    rows_summary: List[Dict[str, Any]] = []
    summary_cols = ["Rank", "Region", "NodesFile", "EdgesFile", "NumberNodes", "NumberEdges"]
    node_cols = ["node_id", "label", "gene_id", "gene_name", "role", "initial_heat", "node_color", "node_size", "node_shape", "path_distance", "path_hops", "gwas_snps", "gwas_disease_traits"]
    edge_cols = ["source", "target", "base_weight", "dynamic_weight", "modifier", "edge_type"]
    if top_network_n <= 0:
        _write_empty_tsv(report_dir / "network_summary.tsv", summary_cols)
        return

    node_names = [str(n) for n in nodes]
    node_to_idx_local = {str(n): int(i) for n, i in node_to_idx.items()}
    n_nodes = len(node_names)
    inputs = report_inputs if isinstance(report_inputs, pd.DataFrame) else top
    heat_with_focal = _raw_seed_heat_from_report_inputs(
        inputs, node_to_idx_local, n_nodes, excluded_focal_idxs=None, fallback_seed_vector=seed_vector
    )

    for rank_idx, (_, row) in enumerate(top.head(top_network_n).iterrows(), start=1):
        region = row.get("input_region", "")
        focal_idxs = sorted({
            int(node_to_idx_local[n])
            for n in _split_semicolon(row.get("overlapped_nodes", ""))
            if n in node_to_idx_local
        })
        if not focal_idxs:
            focal_id = str(row.get("max_rwr_node", "") or "")
            if focal_id in node_to_idx_local:
                focal_idxs = [int(node_to_idx_local[focal_id])]
        focal_set = set(focal_idxs)

        seed_heat_raw = _raw_seed_heat_from_report_inputs(
            inputs, node_to_idx_local, n_nodes, excluded_focal_idxs=focal_set,
            fallback_seed_vector=seed_vector,
        )
        seed_idxs = [int(i) for i in np.flatnonzero(seed_heat_raw > 0) if int(i) not in focal_set]

        paths = _extract_standalone_paths(
            n_nodes=n_nodes,
            u_idx=np.asarray(u_idx, dtype=int),
            v_idx=np.asarray(v_idx, dtype=int),
            edge_weight=np.asarray(dyn_weight, dtype=float),
            focal_idxs=focal_idxs,
            seed_idxs=seed_idxs,
            max_hops=max_hops,
            metric="inverse_weight",
        ) if focal_idxs and seed_idxs else {}

        reachable_seed_idxs = sorted(paths.keys())
        path_node_set, path_pairs = _path_nodes_and_pairs(paths)
        path_node_set.update(focal_idxs)
        path_node_set.update(reachable_seed_idxs)
        edge_records_raw = _edge_records_for_path_pairs(
            path_pairs,
            np.asarray(u_idx, dtype=int), np.asarray(v_idx, dtype=int),
            np.asarray(dyn_weight, dtype=float),
            base_weight=np.asarray(base_weight, dtype=float),
            modifier_by_edge=np.asarray(modifier_by_edge, dtype=float),
        )

        displayed_seed_idxs = sorted(set(reachable_seed_idxs) | {
            i for i in focal_idxs if heat_with_focal[i] > 0
        })
        color_map = _seed_color_from_raw_heat(displayed_seed_idxs, heat_with_focal)
        path_meta = {
            int(si): (float(info["distance"]), len(info["path"]) - 1)
            for si, info in paths.items()
        }

        node_records: List[Dict[str, Any]] = []
        for idx in sorted(path_node_set):
            nid = node_names[idx]
            attrs = dict((node_attributes or {}).get(nid, {}) or {})
            gene_name = attrs.get("gene_name", "")
            gene_id = attrs.get("gene_id", "")
            # Match the validated standalone exporter: gene_id first, ENSG ID fallback.
            is_gene = (
                not _is_missing(gene_id)
                or str(nid).upper().startswith("ENSG")
            )
            label = gene_name if not _is_missing(gene_name) else nid
            is_focal = idx in focal_set
            is_seed = idx in paths
            is_seed_visual = is_seed or (is_focal and heat_with_focal[idx] > 0)
            role = "focal" if is_focal else ("seed" if is_seed else "intermediate")
            heat = float(heat_with_focal[idx]) if is_seed_visual else ""
            dist_val, hops_val = path_meta.get(idx, ("", ""))
            node_records.append({
                "node_id": nid,
                "label": label,
                "gene_id": gene_id,
                "gene_name": gene_name,
                "role": role,
                "initial_heat": heat,
                "node_color": color_map.get(idx, "#D9D9D9") if is_seed_visual else "#D9D9D9",
                "node_size": 42 if (is_focal or is_seed) else 26,
                "node_shape": "ellipse" if is_gene else "triangle",
                "path_distance": dist_val,
                "path_hops": hops_val,
                "gwas_snps": attrs.get("gwas_snps", ""),
                "gwas_disease_traits": attrs.get("gwas_disease_traits", ""),
            })

        edge_records: List[Dict[str, Any]] = []
        for rec in edge_records_raw:
            u_id, v_id = node_names[rec["u_idx"]], node_names[rec["v_idx"]]
            edge_records.append({
                "source": u_id,
                "target": v_id,
                "base_weight": float(rec["base_weight"]),
                "dynamic_weight": float(rec["dynamic_weight"]),
                "modifier": rec.get("modifier", ""),
                "edge_type": "dynamic_network_edge",
            })

        prefix = f"region{rank_idx:03d}"
        nf = network_dir / f"{prefix}_nodes.tsv"
        ef = network_dir / f"{prefix}_edges.tsv"
        pd.DataFrame(node_records, columns=node_cols).to_csv(nf, sep="\t", index=False)
        pd.DataFrame(edge_records, columns=edge_cols).to_csv(ef, sep="\t", index=False)
        rows_summary.append({
            "Rank": rank_idx,
            "Region": region,
            "NodesFile": f"network/{nf.name}",
            "EdgesFile": f"network/{ef.name}",
            "NumberNodes": len(node_records),
            "NumberEdges": len(edge_records),
        })

    pd.DataFrame(rows_summary, columns=summary_cols).to_csv(report_dir / "network_summary.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Report v3: biologically interpretable propagation summaries
# -----------------------------------------------------------------------------

def _normalize_seed_vector(seed_vector: Optional[Sequence[float]], n_nodes: int) -> Optional[np.ndarray]:
    if seed_vector is None:
        return None
    arr = np.asarray(seed_vector, dtype=float).reshape(-1)
    if len(arr) != n_nodes:
        raise ValueError(f"seed_vector length {len(arr)} does not match number of nodes {n_nodes}")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    total = float(arr.sum())
    if total <= 0:
        return None
    return arr / total


def _combine_seed_values(values: Sequence[float], combine_mode: str) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0
    if combine_mode == "mean":
        return float(vals.mean())
    if combine_mode == "union":
        m = float(vals.max())
        if m <= 0:
            return 0.0
        scaled = np.clip(vals / m, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - scaled))
    return float(vals.max())


def _reconstruct_seed_vector_from_ranking(
    ranking: pd.DataFrame,
    node_to_idx: Mapping[str, int],
    n_nodes: int,
    combine_mode: str,
    reference_nodes: Optional[Iterable[str]] = None,
    seed_mode: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Reconstruct the RWR restart vector when the caller does not pass it.

    In Mode2A/2B input-score mode, each region's total score is divided equally
    among its unique overlapped nodes, shared-node contributions are summed, and
    the result is globally normalized.  ``combine_mode`` is retained solely for
    API compatibility and is not used for seed construction.
    """
    has_input_score = "input_score" in ranking.columns and pd.to_numeric(
        ranking["input_score"], errors="coerce"
    ).notna().any()

    if has_input_score and seed_mode != "seed_nodes":
        seed = np.zeros(n_nodes, dtype=float)
        for _, row in ranking.iterrows():
            score = _safe_float(row.get("input_score"), np.nan)
            if not np.isfinite(score):
                continue
            idxs = sorted({
                int(node_to_idx[str(node)])
                for node in _split_semicolon(row.get("overlapped_nodes", ""))
                if str(node) in node_to_idx
            })
            if not idxs:
                continue
            share = float(score) / float(len(idxs))
            for idx in idxs:
                seed[idx] += share
        return _normalize_seed_vector(seed, n_nodes)


    valid_reference = [node_to_idx[str(n)] for n in (reference_nodes or []) if str(n) in node_to_idx]
    if valid_reference:
        seed = np.zeros(n_nodes, dtype=float)
        seed[sorted(set(valid_reference))] = 1.0
        return _normalize_seed_vector(seed, n_nodes)

    # Dynamic binary-input mode can be reconstructed from the input-region overlaps.
    if seed_mode == "input_regions_binary":
        idxs = set()
        for nodes_text in ranking.get("overlapped_nodes", pd.Series(dtype=str)):
            for node in _split_semicolon(nodes_text):
                idx = node_to_idx.get(str(node))
                if idx is not None:
                    idxs.add(int(idx))
        if idxs:
            seed = np.zeros(n_nodes, dtype=float)
            seed[sorted(idxs)] = 1.0
            return _normalize_seed_vector(seed, n_nodes)
    return None


def _static_edge_arrays(G: Any, node_list: Sequence[str], node_to_idx: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    us: List[int] = []
    vs: List[int] = []
    ws: List[float] = []
    if G is None:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    for u, v, attrs in G.edges(data=True):
        su, sv = str(u), str(v)
        if su not in node_to_idx or sv not in node_to_idx:
            continue
        w = _safe_float(attrs.get("weight", 1.0), 1.0)
        if w <= 0:
            continue
        us.append(int(node_to_idx[su])); vs.append(int(node_to_idx[sv])); ws.append(float(w))
    return np.asarray(us, dtype=int), np.asarray(vs, dtype=int), np.asarray(ws, dtype=float)


def _weighted_degree_arrays(n_nodes: int, u_idx: np.ndarray, v_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    deg = np.bincount(u_idx, weights=weights, minlength=n_nodes).astype(float)
    deg += np.bincount(v_idx, weights=weights, minlength=n_nodes).astype(float)
    deg[deg <= 0] = 1.0
    return deg


def _reverse_target_kernel(
    target_idx: int,
    n_nodes: int,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    weights: np.ndarray,
    restart: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Return q such that target RWR score equals dot(seed_vector, q).

    For p = (1-r) P^T p + r s, q solves q = (1-r) P q + r e_target.
    Thus seed-specific contributions to target are s_k * q_k.
    """
    deg = _weighted_degree_arrays(n_nodes, u_idx, v_idx, weights)
    e = np.zeros(n_nodes, dtype=float)
    e[int(target_idx)] = 1.0
    q = e.copy()
    alpha = 1.0 - float(restart)
    for _ in range(int(max_iter)):
        walk = np.zeros(n_nodes, dtype=float)
        # (P q)_u = sum_v w_uv / degree(u) * q_v
        np.add.at(walk, u_idx, weights * q[v_idx] / deg[u_idx])
        np.add.at(walk, v_idx, weights * q[u_idx] / deg[v_idx])
        q_new = alpha * walk + float(restart) * e
        if float(np.abs(q_new - q).sum()) < float(tol):
            q = q_new
            break
        q = q_new
    q[q < 0] = 0.0
    return q


def _adjacency_from_arrays(n_nodes: int, u_idx: np.ndarray, v_idx: np.ndarray) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(u_idx, v_idx):
        ui, vi = int(u), int(v)
        adj[ui].append(vi); adj[vi].append(ui)
    return adj


def _shortest_distances(target_idx: int, adjacency: Sequence[Sequence[int]]) -> Dict[int, int]:
    dist = {int(target_idx): 0}
    queue = deque([int(target_idx)])
    while queue:
        current = queue.popleft()
        for nbr in adjacency[current]:
            if int(nbr) not in dist:
                dist[int(nbr)] = dist[current] + 1
                queue.append(int(nbr))
    return dist


def _attrs_for_node(node_id: str, G: Any = None, node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Mapping[str, Any]:
    if node_attributes is not None and str(node_id) in node_attributes:
        return node_attributes[str(node_id)]
    try:
        if G is not None and node_id in G:
            return dict(G.nodes[node_id])
    except Exception:
        pass
    return {}


def _gene_label_from_attrs(node_id: str, attrs: Mapping[str, Any]) -> Optional[str]:
    """Return a readable gene identifier.

    Priority:
      1. gene_name / gene_symbol / symbol / external_gene_name
      2. gene_id
      3. an explicit non-default label attached to an ENSG node
    """
    for key in ["gene_name", "gene_symbol", "symbol", "external_gene_name"]:
        value = attrs.get(key)
        if not _is_missing(value):
            return str(value)

    gene_id = attrs.get("gene_id")
    if not _is_missing(gene_id):
        return str(gene_id)

    label = attrs.get("label") or attrs.get("name")
    if str(node_id).upper().startswith("ENSG") and not _is_missing(label) and str(label) != str(node_id):
        return str(label)
    return None


def _variant_text(attrs: Mapping[str, Any]) -> str:
    values: List[str] = []
    for key in ["covered_variants", "variants", "variant_ids", "gwas_variants", "GWAS", "gwas"]:
        value = attrs.get(key)
        if _is_missing(value):
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend([str(x) for x in value if not _is_missing(x)])
        else:
            values.extend(_split_semicolon(value) or [str(value)])
    return ";".join(dict.fromkeys(values))


def _gwas_attribute_text(attrs: Mapping[str, Any], key: str) -> str:
    """Return a semicolon-delimited GWAS node attribute."""
    value = attrs.get(key)
    if _is_missing(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(dict.fromkeys(str(x) for x in value if not _is_missing(x)))
    values = _split_semicolon(value)
    return ";".join(dict.fromkeys(values)) if values else str(value)


def _max_scoring_node_for_region(
    row: Mapping[str, Any],
    node_to_idx: Mapping[str, int],
    final_node_scores: np.ndarray,
) -> Optional[Tuple[str, int, float]]:
    """Always return the maximum-final-RWR overlapped node, even when region combine=mean."""
    candidates: List[Tuple[str, int, float]] = []
    for node in _split_semicolon(row.get("overlapped_nodes", "")):
        idx = node_to_idx.get(str(node))
        if idx is None or int(idx) >= len(final_node_scores):
            continue
        score = float(final_node_scores[int(idx)])
        if np.isfinite(score):
            candidates.append((str(node), int(idx), score))
    return max(candidates, key=lambda x: x[2]) if candidates else None



def _covered_gene_summary(
    overlapped_nodes: Sequence[str],
    G: Any = None,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Return genes represented by network nodes directly covered by an input region.

    Only the overlapped nodes themselves are considered. For each overlapped node,
    gene_name is preferred and gene_id is used as fallback. Neighboring genes are
    intentionally excluded.
    """
    genes: List[str] = []

    for node in overlapped_nodes:
        node = str(node)
        attrs = _attrs_for_node(node, G=G, node_attributes=node_attributes)
        label = _gene_label_from_attrs(node, attrs)
        if label:
            genes.append(label)

    return ";".join(dict.fromkeys(genes))


def _gene_name_from_attrs(attrs: Mapping[str, Any]) -> Optional[str]:
    """Return a readable gene name without falling back to a gene identifier.

    Network files have used several synonymous attribute names across versions.
    Only explicit gene-name/symbol attributes are accepted here so enhancer
    neighbors without a readable gene name are not displayed in the report.
    """
    for key in ["gene_name", "gene_symbol", "symbol", "external_gene_name"]:
        value = attrs.get(key)
        if not _is_missing(value):
            return str(value)
    return None


def _region_display_gene(
    overlapped_nodes: Sequence[str],
    nodes: Sequence[str],
    node_to_idx: Mapping[str, int],
    adjacency: Sequence[Sequence[int]],
    G: Any = None,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Return the gene label used in candidate-ranking tables and plots.

    This intentionally reuses the same gene-label extraction routine that was
    already used by the earlier working report fields.  In particular,
    _gene_label_from_attrs() recognizes gene_name/gene_symbol/symbol/
    external_gene_name and retains the existing gene_id fallback.

    Rules:
      1. If the input directly covers one or more gene nodes, show those genes.
      2. Otherwise, show gene-bearing first neighbors of all covered nodes.

    The output is deterministic and semicolon-delimited.
    """
    normalized_overlaps = [str(node) for node in overlapped_nodes]

    covered_genes: List[str] = []
    for node in normalized_overlaps:
        attrs = _attrs_for_node(node, G=G, node_attributes=node_attributes)
        label = _gene_label_from_attrs(node, attrs)
        if label:
            covered_genes.append(label)

    if covered_genes:
        return ";".join(sorted(dict.fromkeys(covered_genes), key=str.casefold))

    neighbor_genes: List[str] = []
    for node in normalized_overlaps:
        idx = node_to_idx.get(node)
        if idx is None or int(idx) >= len(adjacency):
            continue
        for nbr_idx in adjacency[int(idx)]:
            if int(nbr_idx) < 0 or int(nbr_idx) >= len(nodes):
                continue
            nbr_node = str(nodes[int(nbr_idx)])
            attrs = _attrs_for_node(nbr_node, G=G, node_attributes=node_attributes)
            label = _gene_label_from_attrs(nbr_node, attrs)
            if label:
                neighbor_genes.append(label)

    return ";".join(sorted(dict.fromkeys(neighbor_genes), key=str.casefold))


def _direct_gene_summary(
    focal_idx: int,
    overlapped_nodes: Sequence[str],
    nodes: Sequence[str],
    adjacency: Sequence[Sequence[int]],
    G: Any = None,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Return genes directly associated with the max-final-RWR overlapped node.

    The report intentionally uses only:
      - the focal node itself, when it has gene_name or gene_id; and
      - direct neighbors of the focal node with gene_name or gene_id.

    Other overlapped nodes are not included.
    """
    genes: List[str] = []

    focal_node = str(nodes[int(focal_idx)])
    focal_attrs = _attrs_for_node(focal_node, G=G, node_attributes=node_attributes)
    focal_gene = _gene_label_from_attrs(focal_node, focal_attrs)
    if focal_gene:
        genes.append(focal_gene)

    for nbr_idx in adjacency[int(focal_idx)]:
        node = str(nodes[int(nbr_idx)])
        attrs = _attrs_for_node(node, G=G, node_attributes=node_attributes)
        label = _gene_label_from_attrs(node, attrs)
        if label:
            genes.append(label)

    return ";".join(dict.fromkeys(genes))


def _annotate_propagation_explanations(
    ranking: pd.DataFrame,
    nodes: Sequence[str],
    node_to_idx: Mapping[str, int],
    final_node_scores: Sequence[float],
    seed_vector: Optional[Sequence[float]],
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    weights: np.ndarray,
    restart: float,
    tol: float,
    max_iter: int,
    report_top_n: int,
    contribution_fraction_threshold: float,
    G: Any = None,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add max-node, seed-provenance, and direct-gene fields to ranking.

    Exact seed contributions exploit RWR linearity. A seed is counted as "reached"
    when its contribution is at least contribution_fraction_threshold of the focal
    node's total seed-derived RWR score. Distances are shortest-path distances over
    the full weighted-network topology (weights do not affect the distance count).
    """
    out = ranking.copy()
    n_nodes = len(nodes)
    scores = np.asarray(final_node_scores, dtype=float).reshape(-1)
    seed = _normalize_seed_vector(seed_vector, n_nodes)
    adjacency = _adjacency_from_arrays(n_nodes, u_idx, v_idx)
    deg = _weighted_degree_arrays(n_nodes, u_idx, v_idx, weights)
    incident: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for u, v, w in zip(u_idx, v_idx, weights):
        incident[int(u)].append((int(v), float(w)))
        incident[int(v)].append((int(u), float(w)))

    out["max_rwr_node"] = ""
    out["max_rwr_node_label"] = ""
    out["max_rwr_node_score"] = np.nan
    out["n_seeds_reached"] = np.nan
    out["direct_interacting_genes"] = ""
    out["covered_node_genes"] = ""
    out["display_gene"] = ""

    seed_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []
    seed_indices = np.where(seed > 0)[0] if seed is not None else np.array([], dtype=int)

    for row_pos, (_, row) in enumerate(out.iterrows()):
        overlapped_nodes = _split_semicolon(row.get("overlapped_nodes", ""))
        out.at[out.index[row_pos], "display_gene"] = _region_display_gene(
            overlapped_nodes=overlapped_nodes,
            nodes=nodes,
            node_to_idx=node_to_idx,
            adjacency=adjacency,
            G=G,
            node_attributes=node_attributes,
        )

        max_info = _max_scoring_node_for_region(row, node_to_idx, scores)
        if max_info is None:
            continue
        focal_node, focal_idx, focal_score = max_info
        attrs = _attrs_for_node(focal_node, G=G, node_attributes=node_attributes)
        focal_label = _node_display_label(focal_node, G) if G is not None else (
            _gene_label_from_attrs(focal_node, attrs) or str(attrs.get("label", focal_node))
        )
        out.at[out.index[row_pos], "max_rwr_node"] = focal_node
        out.at[out.index[row_pos], "max_rwr_node_label"] = focal_label
        out.at[out.index[row_pos], "max_rwr_node_score"] = focal_score

        # Retain the legacy field for backward compatibility, but the report now
        # displays only genes represented by nodes directly covered by the input.
        out.at[out.index[row_pos], "direct_interacting_genes"] = _direct_gene_summary(
            focal_idx=focal_idx,
            overlapped_nodes=overlapped_nodes,
            nodes=nodes,
            adjacency=adjacency,
            G=G,
            node_attributes=node_attributes,
        )
        out.at[out.index[row_pos], "covered_node_genes"] = _covered_gene_summary(
            overlapped_nodes=overlapped_nodes,
            G=G,
            node_attributes=node_attributes,
        )
        # Contribution decomposition is generated for report candidates only.
        if row_pos >= int(report_top_n) or seed is None or len(seed_indices) == 0:
            continue
        kernel = _reverse_target_kernel(
            target_idx=focal_idx,
            n_nodes=n_nodes,
            u_idx=u_idx,
            v_idx=v_idx,
            weights=weights,
            restart=restart,
            tol=tol,
            max_iter=max_iter,
        )
        contributions = seed * kernel
        total_contribution = float(contributions.sum())
        fractions = contributions / total_contribution if total_contribution > 0 else np.zeros(n_nodes)
        distances = _shortest_distances(focal_idx, adjacency)
        qualifying = seed_indices[fractions[seed_indices] >= float(contribution_fraction_threshold)]
        out.at[out.index[row_pos], "n_seeds_reached"] = int(len(qualifying))

        region = row.get("input_region", "")
        for seed_idx in seed_indices:
            contribution = float(contributions[int(seed_idx)])
            fraction = float(fractions[int(seed_idx)])
            if contribution <= 0:
                continue
            seed_node = str(nodes[int(seed_idx)])
            seed_attrs = _attrs_for_node(seed_node, G=G, node_attributes=node_attributes)
            seed_label = _gene_label_from_attrs(seed_node, seed_attrs) or str(seed_attrs.get("label", seed_node))
            seed_rows.append({
                "region": region,
                "focal_node": focal_node,
                "focal_node_label": focal_label,
                "seed_node": seed_node,
                "seed_region": seed_node,
                "seed_label": seed_label,
                "seed_heat": float(seed[int(seed_idx)]),
                "contribution": contribution,
                "contribution_fraction": fraction,
                "is_counted_seed": int(fraction >= float(contribution_fraction_threshold)),
                "graph_distance": distances.get(int(seed_idx), ""),
                "gwas_snps": _gwas_attribute_text(seed_attrs, "gwas_snps"),
                "gwas_disease_traits": _gwas_attribute_text(seed_attrs, "gwas_disease_traits"),
            })

        # Immediate expected outflow from the focal node to each direct neighbor.
        for nbr_idx, edge_weight in incident.get(int(focal_idx), []):
            nbr_node = str(nodes[int(nbr_idx)])
            nbr_attrs = _attrs_for_node(nbr_node, G=G, node_attributes=node_attributes)
            nbr_label = _gene_label_from_attrs(nbr_node, nbr_attrs) or str(nbr_attrs.get("label", nbr_node))
            transition_probability = float(edge_weight) / float(deg[int(focal_idx)])
            estimated_outflow = (1.0 - float(restart)) * float(focal_score) * transition_probability
            flow_rows.append({
                "region": region,
                "focal_node": focal_node,
                "focal_node_label": focal_label,
                "neighbor_node": nbr_node,
                "neighbor_region": nbr_node,
                "neighbor_label": nbr_label,
                "neighbor_is_seed": int(seed is not None and seed[int(nbr_idx)] > 0),
                "transition_probability": transition_probability,
                "estimated_heat_outflow": estimated_outflow,
                "gwas_snps": _gwas_attribute_text(nbr_attrs, "gwas_snps"),
                "gwas_disease_traits": _gwas_attribute_text(nbr_attrs, "gwas_disease_traits"),
            })

    seed_df = pd.DataFrame(seed_rows, columns=[
        "region", "focal_node", "focal_node_label", "seed_node", "seed_region", "seed_label",
        "seed_heat", "contribution", "contribution_fraction", "is_counted_seed",
        "graph_distance", "gwas_snps", "gwas_disease_traits",
    ])
    flow_df = pd.DataFrame(flow_rows, columns=[
        "region", "focal_node", "focal_node_label", "neighbor_node", "neighbor_region", "neighbor_label",
        "neighbor_is_seed", "transition_probability", "estimated_heat_outflow",
        "gwas_snps", "gwas_disease_traits",
    ])
    return out, seed_df, flow_df


def _write_new_rank_change_outputs(report_dir: Path, ranking: pd.DataFrame, score_col: str, top_n: int = 20) -> None:
    """Compare user input-score rank with final RWR-score rank.

    rank_change = input_rank - final_rank, so positive values indicate movement up.
    Writes all candidates, largest absolute top20, and input-top20/final-top20 union.
    """
    columns = [
        "input_region", "input_score", "input_rank", "final_score", "final_rank",
        "rank_change", "in_input_top20", "in_final_top20", "display_gene",
    ]
    if "input_score" not in ranking.columns or score_col not in ranking.columns:
        for name in ["rank_change.tsv", "rank_change_largest.tsv", "rank_change_top20_union.tsv"]:
            _write_empty_tsv(report_dir / name, columns)
        return
    tmp = _filter_overlapped_regions(ranking).copy()
    input_numeric = pd.to_numeric(tmp["input_score"], errors="coerce")
    final_numeric = pd.to_numeric(tmp[score_col], errors="coerce")
    if input_numeric.notna().sum() == 0:
        for name in ["rank_change.tsv", "rank_change_largest.tsv", "rank_change_top20_union.tsv"]:
            _write_empty_tsv(report_dir / name, columns)
        return
    # Input ranks are calculated only after unmapped regions are removed.
    tmp["input_rank"] = input_numeric.rank(method="dense", ascending=False, na_option="bottom").astype(int)
    # Final rank is lexicographic: final network score first, then input score.
    tmp["final_rank"] = _dense_rank_lexicographic(final_numeric, input_numeric).astype(int)
    tmp["rank_change"] = tmp["input_rank"] - tmp["final_rank"]
    tmp["in_input_top20"] = (tmp["input_rank"] <= int(top_n)).astype(int)
    tmp["in_final_top20"] = (tmp["final_rank"] <= int(top_n)).astype(int)
    out = pd.DataFrame({
        "input_region": tmp["input_region"],
        "input_score": tmp["input_score"],
        "input_rank": tmp["input_rank"],
        "final_score": tmp[score_col],
        "final_rank": tmp["final_rank"],
        "rank_change": tmp["rank_change"],
        "in_input_top20": tmp["in_input_top20"],
        "in_final_top20": tmp["in_final_top20"],
        "display_gene": tmp["display_gene"] if "display_gene" in tmp.columns else "",
    })
    out.to_csv(report_dir / "rank_change.tsv", sep="\t", index=False)
    out.assign(_abs=out["rank_change"].abs()).sort_values(
        ["_abs", "final_rank"], ascending=[False, True]
    ).drop(columns="_abs").head(int(top_n)).to_csv(
        report_dir / "rank_change_largest.tsv", sep="\t", index=False
    )
    out[(out["in_input_top20"] == 1) | (out["in_final_top20"] == 1)].sort_values(
        ["final_rank", "input_rank"]
    ).to_csv(report_dir / "rank_change_top20_union.tsv", sep="\t", index=False)


# Override the former internal/raw rank-change writers.
def _write_rank_change_static(report_dir: Path, ranking: pd.DataFrame, score_col: str) -> None:
    _write_new_rank_change_outputs(report_dir, ranking, score_col, top_n=20)


def _write_rank_change_dynamic(report_dir: Path, ranking: pd.DataFrame, score_col: str) -> None:
    _write_new_rank_change_outputs(report_dir, ranking, score_col, top_n=20)


# -----------------------------------------------------------------------------
# Public entry points used by ranking scripts
# -----------------------------------------------------------------------------

def generate_static_report_outputs(
    out_df: pd.DataFrame,
    G: Any,
    output_tsv: str,
    mode: str,
    args: Any,
    node_list: Optional[Sequence[str]] = None,
    node_scores: Optional[Sequence[float]] = None,
    reference_nodes: Optional[Iterable[str]] = None,
    reference_label: str = "reference",
    report_dir: Optional[str] = None,
    top_n: int = 20,
    top_network_n: int = 3,
    max_neighbors_per_node: int = 25,
    seed_vector: Optional[Sequence[float]] = None,
    contribution_fraction_threshold: float = 0.01,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    report_path = _mkdir(report_dir or (str(Path(output_tsv).with_suffix("")) + ".report"))
    _mkdir(report_path / "network")
    _mkdir(report_path / "figures")

    score_col = _score_col_for_mode(mode, out_df)
    ranking, top = _prepare_ranking(out_df, score_col, top_n)

    if node_list is None and G is not None:
        node_list = [str(n) for n in G.nodes()]
    node_list = list(node_list or [])
    node_to_idx_local = {str(n): i for i, n in enumerate(node_list)}
    final_node_scores = np.asarray(node_scores if node_scores is not None else np.zeros(len(node_list)), dtype=float)
    seed_vec_local = _normalize_seed_vector(seed_vector, len(node_list)) if node_list else None
    if seed_vec_local is None and node_list:
        seed_vec_local = _reconstruct_seed_vector_from_ranking(
            ranking=ranking, node_to_idx=node_to_idx_local, n_nodes=len(node_list),
            combine_mode=getattr(args, "combine_overlaps", "max"),
            reference_nodes=reference_nodes, seed_mode=None,
        )
    su, sv, sw = _static_edge_arrays(G, node_list, node_to_idx_local)
    if node_list and len(final_node_scores) == len(node_list):
        ranking, seed_contrib, direct_flow = _annotate_propagation_explanations(
            ranking=ranking, nodes=node_list, node_to_idx=node_to_idx_local,
            final_node_scores=final_node_scores, seed_vector=seed_vec_local,
            u_idx=su, v_idx=sv, weights=sw,
            restart=float(getattr(args, "restart", 0.5)),
            tol=float(getattr(args, "tol", 1e-7)),
            max_iter=int(getattr(args, "max_iter", 100)),
            report_top_n=max(int(top_n), 20),
            contribution_fraction_threshold=float(contribution_fraction_threshold),
            G=G, node_attributes=node_attributes,
        )
        top = ranking.head(top_n).copy()
        seed_contrib.to_csv(report_path / "seed_contributions.tsv", sep="\t", index=False)
        direct_flow.to_csv(report_path / "direct_heat_flow.tsv", sep="\t", index=False)
    else:
        _write_empty_tsv(report_path / "seed_contributions.tsv", ["region", "focal_node", "seed_node", "contribution", "contribution_fraction", "graph_distance", "gwas_snps", "gwas_disease_traits"])
        _write_empty_tsv(report_path / "direct_heat_flow.tsv", ["region", "focal_node", "neighbor_node", "transition_probability", "estimated_heat_outflow", "gwas_snps", "gwas_disease_traits"])

    n_filtered = _write_filtered_inputs(report_path, out_df)
    ranking.to_csv(report_path / "ranking.tsv", sep="\t", index=False)
    top.to_csv(report_path / "top20.tsv", sep="\t", index=False)

    n_input = len(out_df)
    n_ranked = len(ranking)
    n_overlapped = int((pd.to_numeric(out_df.get("n_overlapped_nodes", 0), errors="coerce") > 0).sum())
    top_region = top.iloc[0].get("input_region", "") if len(top) else ""
    top_score = top.iloc[0].get(score_col, "") if len(top) else ""

    _write_summary(report_path, [
        ("Mode", mode),
        ("Input regions", n_input),
        ("Filtered inputs (no network overlap)", n_filtered),
        ("Ranked regions", n_ranked),
        ("Regions overlapping network", n_overlapped),
        ("Restart probability", getattr(args, "restart", "")),
        ("Combine method", getattr(args, "combine_overlaps", "")),
        ("Network nodes", G.number_of_nodes() if G is not None and hasattr(G, "number_of_nodes") else ""),
        ("Network edges", G.number_of_edges() if G is not None and hasattr(G, "number_of_edges") else ""),
        (f"{reference_label} nodes", len(set(reference_nodes or []))),
        ("Top region", top_region),
        ("Top score", top_score),
    ])

    _write_metadata(report_path, {
        "report_version": "3.0",
        "mode": mode,
        "output_tsv": output_tsv,
        "score_col": score_col,
        "top_n": top_n,
        "top_network_n": top_network_n,
        "max_neighbors_per_node": max_neighbors_per_node,
        "contribution_fraction_threshold": contribution_fraction_threshold,
        "reference_label": reference_label,
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
    })

    node_score_map = None
    if node_list is not None and node_scores is not None:
        node_score_map = {str(n): float(s) for n, s in zip(node_list, node_scores)}

    _write_evidence_summary(report_path, top, score_col, mode, node_score_map=node_score_map, G=G)
    _write_evidence_details(report_path, top, score_col, mode, node_score_map=node_score_map, G=G)
    _write_candidate_cards(report_path, top, score_col, mode, node_score_map=node_score_map, G=G)
    _write_algorithm_trace_static(report_path, top, score_col)
    _write_rank_change_static(report_path, ranking, score_col)

    _export_static_networks(
        report_path, top, G, node_score_map, top_network_n, max_neighbors_per_node,
        seed_vector=seed_vec_local, node_list=node_list, node_attributes=node_attributes,
        max_hops=2, report_inputs=out_df,
    )
    return str(report_path)


def generate_dynamic_report_outputs(
    out_df: pd.DataFrame,
    output_tsv: str,
    args: Any,
    nodes: Sequence[str],
    node_to_idx: Mapping[str, int],
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    base_weight: np.ndarray,
    dyn_weight: np.ndarray,
    edge_idx: np.ndarray,
    modifier: np.ndarray,
    p_adjusted: Optional[np.ndarray] = None,
    celltype: str = "unknown",
    seed_mode: str = "unknown",
    report_dir: Optional[str] = None,
    top_n: int = 20,
    top_network_n: int = 3,
    max_neighbors_per_node: int = 25,
    seed_vector: Optional[Sequence[float]] = None,
    contribution_fraction_threshold: float = 0.01,
    node_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    report_path = _mkdir(report_dir or (str(Path(output_tsv).with_suffix("")) + ".report"))
    _mkdir(report_path / "network")
    _mkdir(report_path / "figures")

    score_col = _score_col_for_mode("dynamic", out_df)
    ranking, top = _prepare_ranking(out_df, score_col, top_n)

    node_names = [str(n) for n in nodes]
    seed_vec_local = _normalize_seed_vector(seed_vector, len(node_names))
    if seed_vec_local is None:
        seed_vec_local = _reconstruct_seed_vector_from_ranking(
            ranking=ranking, node_to_idx=node_to_idx, n_nodes=len(node_names),
            combine_mode=getattr(args, "combine_overlaps", "max"),
            reference_nodes=None, seed_mode=seed_mode,
        )
    final_node_scores = np.asarray(p_adjusted if p_adjusted is not None else np.zeros(len(node_names)), dtype=float)
    if len(final_node_scores) == len(node_names):
        ranking, seed_contrib, direct_flow = _annotate_propagation_explanations(
            ranking=ranking, nodes=node_names, node_to_idx=node_to_idx,
            final_node_scores=final_node_scores, seed_vector=seed_vec_local,
            u_idx=np.asarray(u_idx, dtype=int), v_idx=np.asarray(v_idx, dtype=int),
            weights=np.asarray(dyn_weight, dtype=float),
            restart=float(getattr(args, "restart", 0.5)),
            tol=float(getattr(args, "tol", 1e-9)),
            max_iter=int(getattr(args, "max_iter", 100)),
            report_top_n=max(int(top_n), 20),
            contribution_fraction_threshold=float(contribution_fraction_threshold),
            G=None, node_attributes=node_attributes,
        )
        top = ranking.head(top_n).copy()
        seed_contrib.to_csv(report_path / "seed_contributions.tsv", sep="\t", index=False)
        direct_flow.to_csv(report_path / "direct_heat_flow.tsv", sep="\t", index=False)
    else:
        _write_empty_tsv(report_path / "seed_contributions.tsv", ["region", "focal_node", "seed_node", "contribution", "contribution_fraction", "graph_distance", "gwas_snps", "gwas_disease_traits"])
        _write_empty_tsv(report_path / "direct_heat_flow.tsv", ["region", "focal_node", "neighbor_node", "transition_probability", "estimated_heat_outflow", "gwas_snps", "gwas_disease_traits"])

    n_filtered = _write_filtered_inputs(report_path, out_df)
    ranking.to_csv(report_path / "ranking.tsv", sep="\t", index=False)
    top.to_csv(report_path / "top20.tsv", sep="\t", index=False)

    n_input = len(out_df)
    n_ranked = len(ranking)
    n_overlapped = int((pd.to_numeric(out_df.get("n_overlapped_nodes", 0), errors="coerce") > 0).sum())
    top_region = top.iloc[0].get("input_region", "") if len(top) else ""
    top_score = top.iloc[0].get(score_col, "") if len(top) else ""

    _write_summary(report_path, [
        ("Mode", "celltype_dynamic"),
        ("Cell type", celltype),
        ("Seed mode", seed_mode),
        ("Input regions", n_input),
        ("Filtered inputs (no network overlap)", n_filtered),
        ("Ranked regions", n_ranked),
        ("Regions overlapping network", n_overlapped),
        ("Restart probability", getattr(args, "restart", "")),
        ("Combine method", getattr(args, "combine_overlaps", "")),
        ("Network nodes", len(nodes)),
        ("Network edges", len(base_weight)),
        ("Dynamic override edges", len(edge_idx)),
        ("Minimum component size", getattr(args, "min_component_size", "")),
        ("Coreness penalty mode", getattr(args, "use_coreness_penalty", "")),
        ("Top region", top_region),
        ("Top score", top_score),
    ])

    _write_metadata(report_path, {
        "report_version": "3.0",
        "mode": "celltype_dynamic",
        "celltype": celltype,
        "seed_mode": seed_mode,
        "output_tsv": output_tsv,
        "score_col": score_col,
        "top_n": top_n,
        "top_network_n": top_network_n,
        "max_neighbors_per_node": max_neighbors_per_node,
        "contribution_fraction_threshold": contribution_fraction_threshold,
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
    })

    dynamic_node_score_map = None
    if p_adjusted is not None:
        dynamic_node_score_map = {
            str(n): float(p_adjusted[int(node_to_idx[n])])
            for n in nodes
            if n in node_to_idx and int(node_to_idx[n]) < len(p_adjusted)
        }

    _write_evidence_summary(report_path, top, score_col, "celltype_dynamic", node_score_map=dynamic_node_score_map, G=None)
    _write_evidence_details(report_path, top, score_col, "celltype_dynamic", node_score_map=dynamic_node_score_map, G=None)
    _write_candidate_cards(report_path, top, score_col, "celltype_dynamic", node_score_map=dynamic_node_score_map, G=None)
    _write_algorithm_trace_dynamic(report_path, top, score_col)
    _write_rank_change_dynamic(report_path, ranking, score_col)

    # Edge modifier array aligned to base_weight/dyn_weight. Use existing modifier
    # values for overridden edges and derive fallback from dynamic/base weight.
    modifier_by_edge = np.full(len(base_weight), np.nan, dtype=float)
    try:
        nonzero = base_weight != 0
        modifier_by_edge[nonzero] = dyn_weight[nonzero] / base_weight[nonzero]
    except Exception:
        pass
    try:
        if edge_idx is not None and modifier is not None and len(edge_idx) > 0:
            modifier_by_edge[edge_idx] = modifier
    except Exception:
        pass

    _export_dynamic_networks(
        report_path,
        top,
        nodes,
        node_to_idx,
        u_idx,
        v_idx,
        base_weight,
        dyn_weight,
        modifier_by_edge,
        p_adjusted,
        top_network_n,
        max_neighbors_per_node,
        seed_vector=seed_vec_local,
        node_attributes=node_attributes,
        max_hops=2,
        report_inputs=out_df,
    )
    return str(report_path)
