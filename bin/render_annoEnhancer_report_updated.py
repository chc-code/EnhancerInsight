#!/usr/bin/env python3
"""Render annoEnhancer R Markdown report from Python.

This keeps the report in Rmd (pretty, stable) while letting Python orchestrate the pipeline.

Example:
  python render_annoEnhancer_report.py \
    --rmd report.Rmd \
    --outdir ../test-result \
    --proj K562-CTCF \
    --genome hg19 \
    --out ../test-result/report.html

Defaults:
  --refdir                <parent-of-bindir>/<genome>
  --tfbs-summary          <refdir>/TFBS-summary-<genome>.txt
  --tissue-map-nascent    <refdir>/HACER-tissue.txt
  --tissue-map-sc         <refdir>/dbscATAC-human-tissue-map.txt
  --enhancer-count-file   <refdir>/dbscATAC-enhancer-count.txt
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_INPUTS = {
    "--tfbs-summary": "TFBS summary table",
    "--tissue-map-nascent": "nascent/HACER tissue map",
    "--tissue-map-sc": "single-cell accessibility tissue map",
    "--enhancer-count-file": "dbscATAC enhancer count table",
}


def resolve_optional_inputs(
    genome: str,
    refdir: Path,
    tfbs_summary: Path | None,
    tissue_map_nascent: Path | None,
    tissue_map_sc: Path | None,
    enhancer_count_file: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve optional arguments to their default file paths."""
    if tfbs_summary is None:
        tfbs_summary = refdir / f"TFBS-summary-{genome}.txt"
    if tissue_map_nascent is None:
        tissue_map_nascent = refdir / "HACER-tissue.txt"
    if tissue_map_sc is None:
        tissue_map_sc = refdir / "dbscATAC-human-tissue-map.txt"
    if enhancer_count_file is None:
        enhancer_count_file = refdir / "dbscATAC-enhancer-count.txt"

    return tfbs_summary, tissue_map_nascent, tissue_map_sc, enhancer_count_file


def check_required_file(path: Path, arg_name: str) -> None:
    """Raise a friendly error if a required input file does not exist."""
    if path.exists():
        return

    label = REQUIRED_INPUTS.get(arg_name, "required input file")
    raise FileNotFoundError(
        f"ERROR: Required file not found for {arg_name}\n"
        f"  Description : {label}\n"
        f"  Expected at : {path}\n\n"
        f"Please either:\n"
        f"  1) Make sure the file exists at the default location, or\n"
        f"  2) Provide it explicitly with:\n"
        f"     {arg_name} <path>\n"
    )


def render_rmd(
    rmd_path: Path,
    out_html: Path,
    outdir: Path,
    proj: str,
    genome: str,
    refdir: Path,
    tfbs_summary: Path | None,
    tissue_map_nascent: Path | None,
    tissue_map_sc: Path | None,
    enhancer_count_file: Path | None,
) -> None:
    rmd_path = rmd_path.resolve()
    out_html = out_html.resolve()
    outdir = outdir.resolve()
    refdir = refdir.resolve()

    if not rmd_path.exists():
        raise FileNotFoundError(f"ERROR: Rmd template not found: {rmd_path}")
    if not refdir.exists():
        raise FileNotFoundError(
            f"ERROR: Reference directory not found: {refdir}\n"
            f"You can provide it explicitly with: --refdir <dir>"
        )

    tfbs_summary, tissue_map_nascent, tissue_map_sc, enhancer_count_file = resolve_optional_inputs(
        genome=genome,
        refdir=refdir,
        tfbs_summary=tfbs_summary,
        tissue_map_nascent=tissue_map_nascent,
        tissue_map_sc=tissue_map_sc,
        enhancer_count_file=enhancer_count_file,
    )

    check_required_file(tfbs_summary, "--tfbs-summary")
    check_required_file(tissue_map_nascent, "--tissue-map-nascent")
    check_required_file(tissue_map_sc, "--tissue-map-sc")
    check_required_file(enhancer_count_file, "--enhancer-count-file")

    params = {
        "outdir": str(outdir),
        "proj": proj,
        "genome": genome,
        "refdir": str(refdir),
        "tfbs_summary": str(tfbs_summary.resolve()),
        "tissue_map_nascent": str(tissue_map_nascent.resolve()),
        "tissue_map_sc": str(tissue_map_sc.resolve()),
        "enhancer_count_file": str(enhancer_count_file.resolve()),
    }
    params_json = json.dumps(params, ensure_ascii=False)

    r_code = f"""
    params <- jsonlite::fromJSON({json.dumps(params_json, ensure_ascii=False)})

    rmarkdown::render(
      input = {json.dumps(str(rmd_path), ensure_ascii=False)},
      output_file = {json.dumps(str(out_html), ensure_ascii=False)},
      params = params,
      envir = new.env(parent = globalenv())
    )
    """

    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".R",
        prefix="render_annoEnhancer_",
        dir=str(outdir),
        encoding="utf-8",
        delete=False,
    ) as fh:
        fh.write(r_code)
        r_script_path = Path(fh.name)

    cmd = ["Rscript", str(r_script_path)]
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(
            "ERROR: R Markdown rendering failed. "
            f"Temporary R script kept at: {r_script_path}",
            file=sys.stderr,
        )
        raise e
    else:
        try:
            r_script_path.unlink()
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render annoEnhancer Rmd report with parameters (Python orchestrator)."
    )
    script_dir = Path(__file__).resolve().parent
    default_rmd = script_dir / "report.Rmd"
    ap.add_argument(
        "--rmd",
        default=str(default_rmd),
        metavar="FILE",
        help="Path to the parameterized Rmd template (default: report.Rmd next to this script).",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        metavar="DIR",
        help="Output directory produced by annoEnhancer.",
    )
    ap.add_argument(
        "--proj",
        required=True,
        metavar="STR",
        help="Project prefix (for example: K562-CTCF).",
    )
    ap.add_argument(
        "--genome",
        default="hg19",
        metavar="STR",
        help="Genome build label (for example: hg19, hg38, mm10, mm39). Default: hg19.",
    )
    ap.add_argument(
        "--refdir",
        default=None,
        metavar="DIR",
        help=(
            "Reference directory containing summary tables and tissue maps. "
            "Default: <parent-of-bindir>/<genome>, where bindir is the directory of this script."
        ),
    )
    ap.add_argument(
        "--tfbs-summary",
        default=None,
        metavar="FILE",
        help=(
            "Path to TFBS summary table. "
            "Default: <refdir>/TFBS-summary-<genome>.txt."
        ),
    )
    ap.add_argument(
        "--tissue-map-nascent",
        "--tissue-map",
        dest="tissue_map_nascent",
        default=None,
        metavar="FILE",
        help=(
            "Path to nascent/HACER tissue map. "
            "Backward compatible with the old --tissue-map argument. "
            "Default: <refdir>/HACER-tissue.txt."
        ),
    )
    ap.add_argument(
        "--tissue-map-sc",
        default=None,
        metavar="FILE",
        help=(
            "Path to single-cell accessibility tissue map. "
            "Default: <refdir>/dbscATAC-human-tissue-map.txt."
        ),
    )
    ap.add_argument(
        "--enhancer-count-file",
        default=None,
        metavar="FILE",
        help=(
            "Path to the dbscATAC enhancer count table used by the updated Rmd. "
            "Default: <refdir>/dbscATAC-enhancer-count.txt."
        ),
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="FILE",
        help="Output HTML path.",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    bindir = Path(__file__).resolve().parent
    refdir = Path(args.refdir) if args.refdir else (bindir.parent / args.genome)

    render_rmd(
        rmd_path=Path(args.rmd),
        out_html=Path(args.out),
        outdir=outdir,
        proj=args.proj,
        genome=args.genome,
        refdir=refdir,
        tfbs_summary=Path(args.tfbs_summary) if args.tfbs_summary else None,
        tissue_map_nascent=Path(args.tissue_map_nascent) if args.tissue_map_nascent else None,
        tissue_map_sc=Path(args.tissue_map_sc) if args.tissue_map_sc else None,
        enhancer_count_file=Path(args.enhancer_count_file) if args.enhancer_count_file else None,
    )


if __name__ == "__main__":
    main()


