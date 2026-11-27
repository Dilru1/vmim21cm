#!/usr/bin/env python3
"""Concatenate stage-1 export shards into the flat summaries/ layout."""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs
from sbi.seeding import apply_arm_name

FILES = ("t.npy", "theta.npy", "original_sim_ids.npy", "noise_ids.npy")
CHUNK = 1_000_000


def parse_part(name):
    """'part_00100_00200' -> (100, 200); None if it does not match."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "part":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    apply_arm_name(cfg)
    summ = Path(arm_dirs(cfg)["summaries"])
    total = int(cfg["data"].get("total_nnoise", 0))

    found = []
    for p in sorted(summ.glob("part_*")):
        r = parse_part(p.name)
        if r is not None and p.is_dir():
            found.append((r[0], r[1], p))
    if not found:
        sys.exit(f"no part_<start>_<end> directories under {summ}")
    found.sort(key=lambda x: x[0])

    problems = []
    for f in FILES:
        for _, _, p in found:
            if not (p / f).exists():
                problems.append(f"{p.name} is missing {f} (shard failed or still running?)")
    cursor = found[0][0]
    if cursor != 0:
        problems.append(f"noise coverage starts at {cursor}, not 0")
    for s, e, p in found:
        if s != cursor:
            problems.append(f"gap or overlap: expected shard to start at {cursor}, "
                            f"{p.name} starts at {s}")
        cursor = e
    if total and cursor != total:
        problems.append(f"noise coverage ends at {cursor}, expected total_nnoise={total}")
    if problems:
        sys.exit("cannot merge:\n  - " + "\n  - ".join(problems))

    lens, specs = [], {}
    for f in FILES:
        a = np.load(found[0][2] / f, mmap_mode="r")
        specs[f] = (a.dtype, a.shape[1:])
    for _, _, p in found:
        n = None
        for f in FILES:
            a = np.load(p / f, mmap_mode="r")
            if a.dtype != specs[f][0] or a.shape[1:] != specs[f][1]:
                sys.exit(f"{p.name}/{f} has dtype/shape {a.dtype}{a.shape[1:]}, "
                         f"expected {specs[f][0]}{specs[f][1]} -- shards were "
                         f"written by different code versions")
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                sys.exit(f"{p.name}: {f} has {a.shape[0]} rows, expected {n} "
                         f"-- shard is internally inconsistent, re-run it")
        lens.append(n)

    n_rows = sum(lens)
    print(f"arm      : {cfg['arm_name']}")
    print(f"shards   : {len(found)}  covering noise[0,{cursor})")
    for (s, e, p), n in zip(found, lens):
        print(f"    {p.name:<24} noise[{s},{e})  {n:,} rows")
    print(f"total    : {n_rows:,} rows")
    for f in FILES:
        print(f"    {f:<22} {specs[f][0]} {(n_rows,) + specs[f][1]}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = {}
    for f in FILES:
        dt, tail = specs[f]
        out[f] = np.lib.format.open_memmap(summ / f, mode="w+", dtype=dt,
                                           shape=(n_rows,) + tail)

    off = 0
    for (s, e, p), n in zip(found, lens):
        for f in FILES:
            src = np.load(p / f, mmap_mode="r")
            for lo in range(0, n, CHUNK):
                hi = min(lo + CHUNK, n)
                out[f][off + lo:off + hi] = src[lo:hi]
        off += n
        print(f"  merged {p.name}  ({off:,}/{n_rows:,})", flush=True)
    assert off == n_rows, f"wrote {off} rows, expected {n_rows}"

    for a in out.values():
        a.flush()

    src_scale = found[0][2] / "export_noise_scale.npy"
    if src_scale.exists():
        shutil.copy(src_scale, summ / "export_noise_scale.npy")

    chk = np.load(summ / "t.npy", mmap_mode="r")
    assert chk.shape[0] == n_rows, "flat t.npy row count mismatch after flush"
    sims = np.load(summ / "original_sim_ids.npy", mmap_mode="r")
    n_uniq = len(np.unique(np.asarray(sims[:min(len(sims), 5_000_000)])))
    print(f"\nverified: {n_rows:,} rows, {n_uniq} distinct sim ids in the first "
          f"{min(len(sims), 5_000_000):,}")

    if args.keep_parts:
        print("--keep-parts: shard directories left in place")
    else:
        for _, _, p in found:
            shutil.rmtree(p)
        print(f"removed {len(found)} shard directories")

    print(f"\nmerged -> {summ}")


if __name__ == "__main__":
    main()
