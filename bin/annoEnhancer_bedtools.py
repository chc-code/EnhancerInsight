#!/usr/bin/env python3
"""Python rewrite of annoEnhancer.pl using bedtools for interval operations.

This version keeps the original Perl workflow and output structure as closely as
possible. It intentionally relies on bedtools (plus standard Unix utilities
such as sort, awk, join, and sed) for genomic interval behavior so results stay
consistent with the Perl pipeline.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

DATE = "$Date: 2023-09-09 23:56:37 -0400 (Sat,  9 Sep 2023) $"
AUTHOR = "$Author: Jing Wang <jing.wang@vanderbilt.edu> $"

MAXTHREAD_DEFAULT = 8
MINLINECOUNT_DEFAULT = 100000
SUPPORTED_GENOMES = {"hg19", "hg38"}

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def printerr(*args: object) -> None:
    print(*args, file=sys.stderr)


def error(msg: str, exit_code: Optional[int] = None) -> None:
    now = datetime.now().ctime()
    printerr(f"[{now}] [ERROR] {msg}")
    if exit_code is not None:
        sys.exit(exit_code)


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def quote(path: Path | str) -> str:
    return subprocess.list2cmdline([str(path)])


class Config:
    def __init__(self, args: argparse.Namespace):
        self.queryfile = Path(args.infile).resolve()
        self.thread = args.thread
        self.maxthread = args.maxt
        self.minlinecount = args.minl
        self.genome = args.genome
        self.out_prefix = args.out_prefix
        self.outdir = Path(args.workdir).resolve()
        self.prior = args.prior
        self.cbinding = Path(args.cb).resolve() if args.cb else None

        self.annotation: Optional[Path] = None
        self.tss: Optional[Path] = None
        self.sedb: Optional[Path] = None
        self.dbsuper: Optional[Path] = None
        self.hacer: Optional[Path] = None
        self.atlas: Optional[Path] = None
        self.vista: Optional[Path] = None
        self.raedb: Optional[Path] = None
        self.ensembl: Optional[Path] = None
        self.encode: Optional[Path] = None
        self.nih: Optional[Path] = None
        self.validated_target: Optional[Path] = None
        self.fantom_target: Optional[Path] = None
        self.tf: Optional[Path] = None
        self.gwas: Optional[Path] = None
        self.eqtl: Optional[Path] = None


CFG: Optional[Config] = None


def genome_file(genome: str, filename: str) -> Path:
    return ROOT_DIR / genome / filename


def configure_genome_files(cfg: Config) -> None:
    genome = cfg.genome
    db_dir = ROOT_DIR / genome
    if not db_dir.exists():
        error(f"Please install the databases for {genome} first!", 1)

    cfg.annotation = genome_file(genome, f"{genome}.basic.annotation")
    cfg.tss = genome_file(genome, f"{genome}-tss-by-Gene.txt")
    cfg.sedb = genome_file(genome, f"SEdb-{genome}.bed")
    cfg.dbsuper = genome_file(genome, f"dbSUPER-{genome}.bed")
    cfg.hacer = genome_file(genome, f"HACER-{genome}.bed")
    cfg.atlas = genome_file(genome, f"EnhancerAtlas-{genome}.bed")
    cfg.vista = genome_file(genome, f"VISTA-{genome}.bed")
    cfg.raedb = genome_file(genome, f"RAEdb-{genome}.bed")
    cfg.ensembl = genome_file(genome, f"Ensembl-{genome}.bed")
    cfg.encode = genome_file(genome, f"ENCODE-{genome}.bed")
    cfg.nih = genome_file(genome, f"NIH-{genome}.bed")
    cfg.validated_target = genome_file(genome, f"validated-targets-{genome}.bed")
    cfg.tf = genome_file(genome, f"TF-bindingsites-{genome}.bed")
    cfg.gwas = genome_file(genome, f"GWAS-SNPs-{genome}.bed")
    cfg.eqtl = genome_file(genome, f"eqtl-variants-{genome}.bed")

    if genome in {"hg19", "hg38"}:
        cfg.fantom_target = genome_file(genome, f"human_enhancer_tss_associations-{genome}.bed")
    else:
        cfg.fantom_target = None

    required = [
        cfg.annotation,
        cfg.tss,
        cfg.sedb,
        cfg.dbsuper,
        cfg.hacer,
        cfg.atlas,
        cfg.vista,
        cfg.raedb,
        cfg.ensembl,
        cfg.encode,
        cfg.nih,
        cfg.validated_target,
        cfg.tf,
        cfg.gwas,
        cfg.eqtl,
    ]
    if cfg.fantom_target:
        required.append(cfg.fantom_target)
    for p in required:
        if not p.exists():
            error(f"Required database file not found: {p}", 1)


def check_args(args: argparse.Namespace) -> Config:
    cfg = Config(args)
    if not cfg.outdir.exists():
        error(f"work directory (-w) '{cfg.outdir}' does not exists", 1)
    if not cfg.queryfile.exists():
        error(f"query file in 3 columns bed format (-in) '{cfg.queryfile}' does not exists", 1)
    if cfg.thread is not None:
        if cfg.thread < 1:
            error("number of thread should be integer", 1)
        if cfg.thread > cfg.maxthread:
            error(
                "number of thread is over the maximum allowed threads. Please adjust numer of thread by -t or the maximum number of thread by -maxt",
                1,
            )
    if cfg.cbinding and not cfg.cbinding.exists():
        error(f"binding sites file (-cb) '{cfg.cbinding}' does not exists", 1)
    if cfg.genome not in SUPPORTED_GENOMES:
        error("Please set genome (-m) as hg19 or hg38", 1)

    configure_genome_files(cfg)
    return cfg


def calculate_chunk_line(queryfile: Path, thread: int) -> Tuple[int, int]:
    try:
        result = run_cmd(f"cat {quote(queryfile)} | wc -l")
        line_count = int(result.stdout.strip())
    except Exception:
        with queryfile.open() as fh:
            line_count = sum(1 for _ in fh)
    printerr(f"NOTICE: the queryfile {queryfile} contains {line_count} lines")
    chunk = math.ceil(line_count / thread)
    return line_count, chunk




def perl_split(text: str, sep: str) -> List[str]:
    """Mirror the Perl script's plain split behavior as closely as possible.

    The original Perl code uses split(/;/, ...), split(/,/, ...), split(/:/, ...),
    which keeps internal empty fields and only drops trailing empties under Perl's
    default split rules. For this pipeline, the safest behavior is to avoid any
    extra filtering and let downstream logic match the Perl code structure as-is.
    """
    return text.split(sep)


def combine_simple(file1: Path, file2: Path) -> None:
    links: Dict[str, str] = {}
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) < 4:
                continue
            key = "\t".join(temp[:3])
            val = temp[3]
            links[key] = f"{links[key]};{val}" if key in links else val
    with file2.open("w") as out:
        for key, val in links.items():
            out.write(f"{key}\t{val}\n")


def combine2(file1: Path, file2: Path) -> None:
    links: Dict[str, object] = {}
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) < 5:
                continue
            key = "\t".join(temp[:4])
            if temp[4] != "NA":
                temp1 = temp[4].split(":")
                source = temp1[0]
                genes = temp1[1] if len(temp1) > 1 else ""
                gene_map = links.setdefault(key, defaultdict(dict))
                assert isinstance(gene_map, defaultdict)
                for gene in genes.split(";"):
                    gene = gene.strip()
                    if not gene:
                        continue
                    gene_map[source][gene] = 1
            else:
                links[key] = "NA"
    with file2.open("w", newline="\n") as out:
        for key, val in links.items():
            out.write(f"{key}\t")
            if val == "NA":
                out.write("NA\n")
                continue
            assert isinstance(val, defaultdict)
            for k1, sub in val.items():
                out.write(f"{k1}:")
                for k2 in sub.keys():
                    if not k2:
                        continue
                    out.write(f"{k2},")
                out.write(";")
            out.write("\n")


def combine3(file1: Path, file2: Path) -> None:
    links: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) >= 2:
                links[temp[0]][temp[1]] = 1
    with file2.open("w") as out:
        for key, sub in links.items():
            out.write(f"{key}\t")
            for key1 in sub.keys():
                out.write(key1 if key1 == "NA" else f"{key1},")
            out.write("\n")


def combine4(file1: Path, file2: Path) -> None:
    links: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) < 2:
                continue
            for item in temp[1].split(","):
                links[temp[0]][item] = 1
    with file2.open("w") as out:
        for key, sub in links.items():
            out.write(f"{key}\t")
            for key1 in sub.keys():
                out.write(key1 if key1 == "NA" else f"{key1},")
            out.write("\n")


def combine5(file1: Path, file2: Path) -> None:
    links: Dict[str, object] = {}
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) < 4:
                continue
            key = "\t".join(temp[:3])
            if temp[3] != "NA":
                temp1 = temp[3].split(":")
                group = links.setdefault(key, defaultdict(dict))
                assert isinstance(group, defaultdict)
                left = temp1[0] if len(temp1) > 0 else ""
                right = temp1[1] if len(temp1) > 1 else ""
                group[left][right] = 1
            else:
                links[key] = "NA"
    with file2.open("w") as out:
        for key, val in links.items():
            out.write(f"{key}\t")
            if val == "NA":
                out.write("NA\n")
                continue
            assert isinstance(val, defaultdict)
            for k1, sub in val.items():
                out.write(f"{k1}:")
                for k2 in sub.keys():
                    out.write(f"{k2},")
                out.write(";")
            out.write("\n")


def combine6(file1: Path, file2: Path) -> None:
    links: Dict[str, object] = {}
    with file1.open() as fh:
        for line in fh:
            temp = line.rstrip("\n").split("\t")
            if len(temp) < 4:
                continue
            key = "\t".join(temp[:3])
            if temp[3] != "NA":
                idx = temp[3].rfind(":")
                left = temp[3][:idx]
                right = temp[3][idx + 1 :]
                group = links.setdefault(key, defaultdict(dict))
                assert isinstance(group, defaultdict)
                group[left][right] = 1
            else:
                links[key] = "NA"
    with file2.open("w") as out:
        for key, val in links.items():
            out.write(f"{key}\t")
            if val == "NA":
                out.write("NA\n")
                continue
            assert isinstance(val, defaultdict)
            for k1, sub in val.items():
                out.write(f"{k1}:")
                for k2 in sub.keys():
                    out.write(f"{k2},")
                out.write(";")
            out.write("\n")


def format_annotation(file: Path) -> None:
    output: List[str] = []
    with file.open() as fh:
        for line in fh:
            temp = line.split("\t")
            if len(temp) < 4:
                output.append(line)
                continue
            value = temp[3]
            if "promoter-TSS" in value:
                temp[3] = "promoter-TSS\n"
            elif "5' UTR" in value:
                temp[3] = "5' UTR\n"
            elif "exon (" in value:
                temp[3] = "exon\n"
            elif "intron (" in value:
                temp[3] = "intron\n"
            elif "non-coding" in value:
                temp[3] = "non-coding\n"
            elif "TTS" in value:
                temp[3] = "TTS\n"
            elif "3' UTR" in value:
                temp[3] = "3' UTR\n"
            elif "Intergenic" in value:
                temp[3] = "intergenic\n"
            output.append("\t".join(temp))
    file.write_text("".join(output))


def annotation(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.annotation
    run_cmd(f"sort -k1,1 -k2,2n {quote(file1)} > {quote(str(file1) + '.sorted')}")
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.annotation)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{print $1,$2,$3,$9}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 -k2,2n > {quote(str(file1) + '.closest.sorted')}"
    )
    combine_simple(Path(f"{file1}.closest.sorted"), file2)
    format_annotation(file2)


def neighbor_gene(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.tss
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.tss)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{print $1,$2,$3,$7}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 -k2,2n > {quote(str(file1) + '.closest.sorted')}"
    )
    combine_simple(Path(f"{file1}.closest.sorted"), file2)


def neighbor_gene_50k(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.tss
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(str(cfg.tss) + '.50k')} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1,$2,$3,$7}}else{{print $1,$2,$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 -k2,2n > {quote(str(file1) + '.closest.sorted')}"
    )
    combine_simple(Path(f"{file1}.closest.sorted"), file2)


def support(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.sedb)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,\"Y\"}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"N\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.SEdb')}"
    )

    datasets = [
        (cfg.dbsuper, "dbSUPER"),
        (cfg.hacer, "HACER"),
        (cfg.atlas, "EnhancerAtlas"),
        (cfg.vista, "VISTA"),
        (cfg.raedb, "RAEdb"),
        (cfg.ensembl, "Ensembl"),
        (cfg.encode, "ENCODE"),
        (cfg.nih, "NIH"),
    ]
    for bed, suffix in datasets:
        assert bed is not None
        run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(bed)} -d > {quote(str(file1) + '.closest')}")
        if suffix in {"Ensembl", "ENCODE"}:
            run_cmd(
                f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,\"Y\"}}else{{print $1\":\"$2\"-\"$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.' + suffix)}"
            )
        elif suffix == "VISTA":
            run_cmd(
                f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$7}}else{{print $1\":\"$2\"-\"$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.VISTA')}"
            )
        else:
            run_cmd(
                f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$7}}else{{print $1\":\"$2\"-\"$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
            )
            combine4(Path(f"{file1}.closest.sorted"), Path(f"{file1}.{suffix}"))

    for suffix in ["SEdb", "dbSUPER", "HACER", "VISTA", "RAEdb", "Ensembl", "ENCODE", "NIH"]:
        run_cmd(f"sort -k1,1 {quote(str(file1) + '.' + suffix)} > {quote(str(file1) + '.' + suffix + '.sorted')}")

    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.SEdb.sorted')} {quote(str(file1) + '.dbSUPER.sorted')} > {quote(str(file1) + '.temp1.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp1.bed')} {quote(str(file1) + '.HACER.sorted')} > {quote(str(file1) + '.temp2.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp2.bed')} {quote(str(file1) + '.VISTA.sorted')} > {quote(str(file1) + '.temp3.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp3.bed')} {quote(str(file1) + '.RAEdb.sorted')} > {quote(str(file1) + '.temp4.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp4.bed')} {quote(str(file1) + '.Ensembl.sorted')} > {quote(str(file1) + '.temp5.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp5.bed')} {quote(str(file1) + '.ENCODE.sorted')} > {quote(str(file1) + '.temp6.bed')}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.temp6.bed')} {quote(str(file1) + '.NIH.sorted')} > {quote(str(file1) + '.temp7.bed')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}}' {quote(str(file1) + '.temp7.bed')} | sort -k1,1 -k2,2n > {quote(file2)}"
    )


def target_gene(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.validated_target
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.validated_target)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,$8\":\"$7}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
    )
    combine2(Path(f"{file1}.closest.sorted"), Path(f"{file1}.target1"))

    if cfg.genome in {"hg19", "hg38"} and cfg.fantom_target:
        run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.fantom_target)} -d > {quote(str(file1) + '.closest')}")
        run_cmd(
            f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$7}}else{{print $1\":\"$2\"-\"$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
        )
        combine3(Path(f"{file1}.closest.sorted"), Path(f"{file1}.target2"))
        run_cmd(f"sort -k1,1 {quote(str(file1) + '.target1')} > {quote(str(file1) + '.target1.sorted')}")
        run_cmd(f"sort -k1,1 {quote(str(file1) + '.target2')} > {quote(str(file1) + '.target2.sorted')}")
        run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(file1) + '.target1.sorted')} {quote(str(file1) + '.target2.sorted')} | cut -f2- > {quote(file2)}")
    else:
        run_cmd(f"sort -k1,1 {quote(str(file1) + '.target1')} > {quote(str(file1) + '.target1.sorted')}")
        run_cmd(f"cut -f2- {quote(str(file1) + '.target1.sorted')} > {quote(file2)}")


def tf_binding(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.tf
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.tf)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,$8\":\"$7\"%\"$9\"%\"$10\"%\"$11}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
    )
    run_cmd(f"cut -f2- {quote(str(file1) + '.closest.sorted')} > {quote(str(file1) + '.closest.sorted.1')}")
    combine5(Path(f"{file1}.closest.sorted.1"), file2)


def cbinding(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.cbinding
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b <(sort -k1,1 -k2,2n {quote(cfg.cbinding)}) -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,\"Y\"}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"N\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(file2)}"
    )


def gwas_snp(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.gwas
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.gwas)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,$7\":\"$9}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
    )
    run_cmd(f"cut -f2- {quote(str(file1) + '.closest.sorted')} > {quote(str(file1) + '.closest.sorted.1')}")
    combine6(Path(f"{file1}.closest.sorted.1"), file2)


def eqtl_variant(file1: Path, file2: Path) -> None:
    cfg = CFG
    assert cfg and cfg.eqtl
    run_cmd(f"bedtools closest -a {quote(str(file1) + '.sorted')} -b {quote(cfg.eqtl)} -d > {quote(str(file1) + '.closest')}")
    run_cmd(
        f"awk -F'\t' -v OFS='\t' '{{if($NF==0){{print $1\":\"$2\"-\"$3,$1,$2,$3,$18\":\"$7\"(\"$8\")-\"$10}}else{{print $1\":\"$2\"-\"$3,$1,$2,$3,\"NA\"}}}}' {quote(str(file1) + '.closest')} | sort | uniq | sort -k1,1 > {quote(str(file1) + '.closest.sorted')}"
    )
    run_cmd(f"cut -f2- {quote(str(file1) + '.closest.sorted')} > {quote(str(file1) + '.closest.sorted.1')}")
    combine5(Path(f"{file1}.closest.sorted.1"), file2)


def annotate_query_by_region_thread(regionout_file: Path, start_line: Optional[int] = None, end_line: Optional[int] = None, cur_thread: Optional[int] = None) -> None:
    cfg = CFG
    assert cfg
    if cfg.thread and start_line is not None and end_line is not None and cur_thread is not None:
        query_path = Path(f"{regionout_file}.{cur_thread}.query")
        run_cmd(f"sed -n {start_line},{end_line}p {quote(cfg.queryfile)} > {quote(query_path)}")
        prefix = Path(f"{regionout_file}.{cur_thread}")
        temp_prefix = cfg.outdir / str(cur_thread)
    else:
        query_path = Path(f"{regionout_file}.query")
        shutil.copyfile(cfg.queryfile, query_path)
        prefix = regionout_file
        temp_prefix = cfg.outdir / ""

    out_annotation = Path(f"{prefix}.annotation")
    out_neighborGene = Path(f"{prefix}.neighborGene")
    out_neighbor50k = Path(f"{prefix}.neighbor50k")
    out_support = Path(f"{prefix}.support")
    out_targetGene = Path(f"{prefix}.targetGene")
    out_TFbinding = Path(f"{prefix}.TFbinding")
    out_cbinding = Path(f"{prefix}.cbinding")
    out_GWASsnp = Path(f"{prefix}.GWASsnp")
    out_eQTLvariant = Path(f"{prefix}.eQTLvariant")

    annotation(query_path, out_annotation)
    neighbor_gene(query_path, out_neighborGene)
    neighbor_gene_50k(query_path, out_neighbor50k)
    support(query_path, out_support)
    target_gene(query_path, out_targetGene)
    tf_binding(query_path, out_TFbinding)
    gwas_snp(query_path, out_GWASsnp)
    eqtl_variant(query_path, out_eQTLvariant)

    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$0}}' {quote(out_annotation)} | sort -k1,1 > {quote(str(out_annotation) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_neighborGene)} | sort -k1,1 > {quote(str(out_neighborGene) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_neighbor50k)} | sort -k1,1 > {quote(str(out_neighbor50k) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4,$5,$6,$7,$8,$9,$10,$11}}' {quote(out_support)} | sort -k1,1 > {quote(str(out_support) + '.wid')}")
    if cfg.genome in {"hg19", "hg38"}:
        run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4,$5}}' {quote(out_targetGene)} | sort -k1,1 > {quote(str(out_targetGene) + '.wid')}")
    else:
        run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_targetGene)} | sort -k1,1 > {quote(str(out_targetGene) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_TFbinding)} | sort -k1,1 > {quote(str(out_TFbinding) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_GWASsnp)} | sort -k1,1 > {quote(str(out_GWASsnp) + '.wid')}")
    run_cmd(f"awk -F'\t' -v OFS='\t' '{{print $1\":\"$2\"-\"$3,$4}}' {quote(out_eQTLvariant)} | sort -k1,1 > {quote(str(out_eQTLvariant) + '.wid')}")

    base = cfg.outdir / (str(cur_thread) if cur_thread is not None else "")
    temp1 = Path(f"{base}.temp1.bed")
    temp2 = Path(f"{base}.temp2.bed")
    temp3 = Path(f"{base}.temp3.bed")
    temp4 = Path(f"{base}.temp4.bed")
    temp5 = Path(f"{base}.temp5.bed")
    temp6 = Path(f"{base}.temp6.bed")
    temp7 = Path(f"{base}.temp7.bed")
    temp8 = Path(f"{base}.temp8.bed")

    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(str(out_annotation) + '.wid')} {quote(str(out_neighborGene) + '.wid')} > {quote(temp1)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp1)} {quote(str(out_neighbor50k) + '.wid')} > {quote(temp2)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp2)} {quote(str(out_support) + '.wid')} > {quote(temp3)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp3)} {quote(str(out_targetGene) + '.wid')} > {quote(temp4)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp4)} {quote(str(out_TFbinding) + '.wid')} > {quote(temp5)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp5)} {quote(str(out_GWASsnp) + '.wid')} > {quote(temp6)}")
    run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp6)} {quote(str(out_eQTLvariant) + '.wid')} > {quote(temp7)}")

    final_annotation = Path(f"{prefix}.annotation.txt")
    if cfg.cbinding:
        cbinding(query_path, out_cbinding)
        run_cmd(f"join -1 1 -2 1 -t $'\t' {quote(temp7)} {quote(out_cbinding)} > {quote(temp8)}")
        run_cmd(f"cut -f2- {quote(temp8)} | sort -k1,1 -k2,2n > {quote(final_annotation)}")
    else:
        run_cmd(f"cut -f2- {quote(temp7)} | sort -k1,1 -k2,2n > {quote(final_annotation)}")

    for f in cfg.outdir.glob("*.temp*.bed"):
        f.unlink(missing_ok=True)


def combine_file(outfile: Path, thread: int) -> None:
    cfg = CFG
    assert cfg
    with outfile.open("w") as out:
        for j in range(thread):
            part = cfg.outdir / f"{cfg.out_prefix}.{j}.annotation.txt"
            if part.exists():
                with part.open() as fh:
                    shutil.copyfileobj(fh, out)
                part.unlink(missing_ok=True)


def add_header(annotation_file: Path) -> None:
    cfg = CFG
    assert cfg
    if cfg.genome in {"hg19", "hg38"}:
        cols = [
            "chr", "start", "end", "Function-NCBI_RefSeq", "NearestGene", "Genes-within-50kb",
            "superEnhancer-SEdb2.0", "superEnhancer-dbSUPER", "Enhancer-HACER", "Enhancer-VISTA",
            "Enhancer-RAEdb", "Enhancer-EnsemblRegulatoryBuild", "Enhancer-ENCODEBroad-ChromHMM",
            "Enhancer-NIH-ROADMAP", "targetGene-experimental", "targetGene-FANTOM5", "TF binding sites",
            "GWAS SNPs", "eQTL variants"
        ]
    else:
        cols = [
            "chr", "start", "end", "Function-NCBI_RefSeq", "NearestGene", "Genes-within-50kb",
            "superEnhancer-SEdb2.0", "superEnhancer-dbSUPER", "Enhancer-HACER", "Enhancer-VISTA",
            "Enhancer-RAEdb", "Enhancer-EnsemblRegulatoryBuild", "Enhancer-ENCODEBroad-ChromHMM",
            "Enhancer-NIH-ROADMAP", "targetGene-experimental", "TF binding sites", "GWAS SNPs", "eQTL variants"
        ]
    if cfg.cbinding:
        cols.append("user_binding")
    original = annotation_file.read_text()
    annotation_file.write_text("\t".join(cols) + "\n" + original)


def apply_prioritization(annotation_file: Path) -> None:
    lines = annotation_file.read_text().splitlines()
    if not lines:
        return
    head, data = lines[0], lines[1:]
    out = [head]
    for row in data:
        temp = row.split("\t")
        score = 0
        for value in temp[6:]:
            if value not in {"NA", "N"}:
                score += 1
        out.append(f"{row}\t{score}")
    annotation_file.write_text("\n".join(out) + "\n")


def postprocess_outputs() -> None:
    cfg = CFG
    assert cfg
    outdir = cfg.outdir
    out_prefix = cfg.out_prefix
    query_sorted = outdir / "queryfile.sorted"
    run_cmd(f"sort -k1,1 -k2,2n {quote(cfg.queryfile)} > {quote(query_sorted)}")

    if cfg.genome in {"hg19", "hg38"}:
        hacer_d = genome_file(cfg.genome, f"HACER-{cfg.genome}-density-normalized.bed")
        closest = outdir / "queryfile.hacer.closest"
        run_cmd(f"bedtools closest -a {quote(query_sorted)} -b {quote(hacer_d)} -d > {quote(closest)}")
        out_hacer = outdir / f"{out_prefix}-HACER.txt"
        run_cmd(
            f"awk -F'\t' -v OFS='\t' '($NF==0){{print $1\":\"$2\"-\"$3,$7,$8}}' {quote(closest)} | sort | uniq > {quote(out_hacer)}"
        )
        scatac_d = genome_file(cfg.genome, f"dbscATAC-{cfg.genome}-enhancer.bed")
        sc_closest = outdir / "queryfile.sc.closest"
        run_cmd(f"bedtools closest -a {quote(query_sorted)} -b {quote(scatac_d)} -d > {quote(sc_closest)}")
        out_sc = outdir / f"{out_prefix}-scATAC.txt"
        run_cmd(
            f"awk -F'\t' -v OFS='\t' '($NF==0){{print $1\":\"$2\"-\"$3,$7,$8}}' {quote(sc_closest)} | sort | uniq > {quote(out_sc)}"
        )

    file = outdir / f"{out_prefix}.annotation.txt"
    c_tf: DefaultDict[str, DefaultDict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    c_snp: Dict[str, int] = {}
    c_eqtl: Dict[str, int] = {}
    t_c: Dict[str, int] = {}
    t_50k: Dict[str, int] = {}
    t_f: Dict[str, int] = {}
    t_e: Dict[str, int] = {}
    t_g: Dict[str, int] = {}

    with file.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if line_no == 1:
                continue
            temp1 = line.rstrip("\n").split("\t")
            if len(temp1) < 19:
                continue

            for item in perl_split(temp1[16], ";"):
                if temp1[16] == "NA":
                    break
                temp3 = item.split(":")
                if len(temp3) != 2:
                    continue
                cellline = temp3[0].strip()
                if not cellline:
                    continue
                for tf_item in perl_split(temp3[1], ","):
                    tf_item = tf_item.strip()
                    if not tf_item:
                        continue
                    cell_bucket = c_tf.setdefault(temp1[3], {}).setdefault(cellline, {})
                    cell_bucket[tf_item] = cell_bucket.get(tf_item, 0) + 1

            for item in perl_split(temp1[17], ";"):
                if temp1[17] == "NA":
                    break
                idx = item.rfind(":")
                if idx == -1:
                    continue
                key = item[:idx].strip()
                if not key:
                    continue
                seen_items = set()
                for x in perl_split(item[idx + 1 :], ","):
                    x = x.strip()
                    if not x or x in seen_items:
                        continue
                    seen_items.add(x)
                    c_snp[key] = c_snp.get(key, 0) + 1

            temp7 = perl_split(temp1[18], ";")
            for item in temp7:
                if temp1[18] == "NA":
                    break
                idx = item.find(":")
                if idx == -1:
                    continue
                tissue = item[:idx].strip()
                if not tissue:
                    continue
                seen_items = set()
                for x in perl_split(item[idx + 1 :], ","):
                    x = x.strip()
                    if not x or x in seen_items:
                        continue
                    seen_items.add(x)
                    c_eqtl[tissue] = c_eqtl.get(tissue, 0) + 1

            for x in perl_split(temp1[4], ";"):
                t_c[x] = 1
            for x in perl_split(temp1[5], ";"):
                t_50k[x] = 1
            for x in perl_split(temp1[15], ","):
                t_f[x] = 1
            for item in temp1[14].split(";"):
                if temp1[14] == "NA":
                    break
                temp5 = item.split(":")
                if len(temp5) != 2:
                    continue
                for g in perl_split(temp5[1], ","):
                    t_e[g] = 1
            for item in temp7:
                if temp1[18] == "NA":
                    break
                temp8 = item.split(":")
                if len(temp8) != 2:
                    continue
                for g in perl_split(temp8[1], ","):
                    idx = g.find("-")
                    if idx != -1:
                        t_g[g[idx + 1 :]] = 1

    with (outdir / f"{out_prefix}.m_tf.txt").open("w", newline="\n") as out:
        for key1, sub1 in c_tf.items():
            for key2, sub2 in sub1.items():
                for key3, count in sub2.items():
                    out.write(f"{key1}\t{key2}\t{key3.replace('%', '::')}\t{count}\n")
    with (outdir / f"{out_prefix}.m_SNP.txt").open("w") as out:
        for k, v in sorted(c_snp.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"{k}\t{v}\n")
    with (outdir / f"{out_prefix}.m_eQTL.txt").open("w") as out:
        for k, v in sorted(c_eqtl.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"{k}\t{v}\n")
    with (outdir / f"{out_prefix}.target.txt").open("w", newline="\n") as out:
        out.write("\t".join(t_c.keys()) + "\n")
        out.write("\t".join(t_50k.keys()) + "\n")
        out.write("\t".join(t_f.keys()) + "\n")
        out.write("\t".join(t_e.keys()) + "\n")
        out.write("\t".join(t_g.keys()) + "\n")

    for f in outdir.glob(f"{out_prefix}.?.*"):
        f.unlink(missing_ok=True)



def cleanup_intermediate_files() -> None:
    cfg = CFG
    assert cfg
    outdir = cfg.outdir
    out_prefix = cfg.out_prefix

    keep = {
        f"{out_prefix}.annotation.txt",
        f"{out_prefix}-HACER.txt",
        f"{out_prefix}-scATAC.txt",
        f"{out_prefix}.m_eQTL.txt",
        f"{out_prefix}.m_SNP.txt",
        f"{out_prefix}.m_tf.txt",
        f"{out_prefix}.target.txt",
    }

    for p in outdir.iterdir():
        if not p.is_file():
            continue
        # only touch files from the current run/prefix
        if not (p.name.startswith(f"{out_prefix}.") or p.name.startswith(f"{out_prefix}-")):
            continue
        if p.name in keep:
            continue
        p.unlink(missing_ok=True)

    # also remove common temp files not prefixed by out_prefix
    for extra in ("queryfile.sorted", "queryfile.hacer.closest", "queryfile.sc.closest", f"{out_prefix}-scATAC.tmp"):
        q = outdir / extra
        q.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python rewrite of annoEnhancer.pl",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-m", dest="genome", default="hg19", help="define the genome: hg19, hg38, mm10, mm39. default: hg19")
    parser.add_argument("-t", dest="thread", type=int, help="define the number of thread")
    parser.add_argument("-maxt", dest="maxt", type=int, default=MAXTHREAD_DEFAULT, help="define the maximum number of thread, default: 8")
    parser.add_argument("-minl", dest="minl", type=int, default=MINLINECOUNT_DEFAULT, help="define the minimum number of input regions/lines")
    parser.add_argument("-in", dest="infile", required=True, help="required, region file in 3 columns bed format (chr start end)")
    parser.add_argument("-o", dest="out_prefix", default="annoEnhancer", help="name prefix for output file")
    parser.add_argument("-w", dest="workdir", required=True, help="required, work directory")
    parser.add_argument("-cb", dest="cb", help="the path to the binding sites in bed format")
    parser.add_argument("-pri", dest="prior", type=int, default=0, help="prioritize annotated enhancers or not, default 0")

    args, unknown = parser.parse_known_args()
    if unknown:
        error(f"Unknown arguments: {' '.join(unknown)}", 1)
    global CFG
    CFG = check_args(args)
    cfg = CFG

    outfile = cfg.outdir / f"{cfg.out_prefix}.annotation.txt"
    queryfile_line_count = 0
    chunk_line_count = 0

    thread = cfg.thread
    if thread:
        queryfile_line_count, chunk_line_count = calculate_chunk_line(cfg.queryfile, thread)
        if queryfile_line_count < cfg.minlinecount:
            printerr(f"NOTICE: threading is disabled for gene-based annotation on file with less than {cfg.minlinecount} input lines")
            thread = None
        if thread and thread > cfg.maxthread:
            printerr(f"NOTICE: number of threads is reduced to {cfg.maxthread} (use --maxthread to change this behavior)")
            thread = cfg.maxthread
            queryfile_line_count, chunk_line_count = calculate_chunk_line(cfg.queryfile, thread)

    if thread:
        futures = []
        with ThreadPoolExecutor(max_workers=thread) as ex:
            for i in range(thread):
                start_line = i * chunk_line_count + 1
                end_line = min(start_line + chunk_line_count - 1, queryfile_line_count)
                printerr(f"NOTICE: Creating new threads for query line {start_line} to {end_line}")
                futures.append(ex.submit(annotate_query_by_region_thread, cfg.outdir / cfg.out_prefix, start_line, end_line, i))
            for fut in futures:
                fut.result()
        combine_file(outfile, thread)
    else:
        annotate_query_by_region_thread(cfg.outdir / cfg.out_prefix)

    add_header(outfile)
    if cfg.prior == 1:
        print("Prioritizing...")
        apply_prioritization(outfile)
    postprocess_outputs()
    cleanup_intermediate_files()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc), 1)
