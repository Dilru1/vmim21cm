#!/usr/bin/env python3
"""Stage 4: evaluate and overplot chain directories listed in .list files."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_list(path):
    """Read one .list file -> [(name, path, n_params|None, [subdir, ...]), ...]."""
    out = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            sys.exit(f"{path}:{lineno}: need at least 'name, path' -- got {raw!r}")
        name, p = parts[0], parts[1]
        npar = int(parts[2]) if len(parts) > 2 and parts[2] else None
        subs = [x for x in parts[3:] if x]
        out.append((name, p, npar, subs))
    if not out:
        sys.exit(f"{path}: no entries")
    return out


def has_dat(d):
    return next(Path(d).glob("*.dat"), None) is not None


def list_chain_dirs(root, name):
    """Every directory under `root` (or root/chains) that holds *.dat."""
    root = Path(root)
    search = root / "chains" if (root / "chains").is_dir() else root
    found = []
    if has_dat(search):
        found.append(search)
    if search.is_dir():
        found += [d for d in sorted(search.iterdir()) if d.is_dir() and has_dat(d)]
    return found


def resolve_chains_dir(root, subdir, name):
    """Find the directory that actually holds the *.dat files."""
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"[{name}] not a directory: {root}")

    if has_dat(root) and not subdir:
        return root

    cands = list_chain_dirs(root, name)

    if subdir:
        cands = [d for d in cands if subdir.lower() in d.name.lower()]
        if not cands:
            sys.exit(f"[{name}] no chains dir under {root} matching {subdir!r}")

    if not cands:
        sys.exit(f"[{name}] no *.dat found under {root} or {root}/chains/*/")
    if len(cands) > 1:
        opts = "\n    ".join(str(c.relative_to(root)) for c in cands)
        sys.exit(f"[{name}] {len(cands)} candidate chain dirs under {root}:\n"
                 f"    {opts}\n"
                 f"  Add a 4th column to the .list line to pick one, e.g.\n"
                 f"    {name}, {root}, , {cands[0].name}")
    return cands[0]


def infer_n_params(chains_dir, name):
    """Width of the chains, taken from the first *_truth.npy."""
    for dat in sorted(Path(chains_dir).glob("*.dat")):
        tp = dat.with_name(dat.stem + "_truth.npy")
        if tp.exists():
            n = int(np.load(tp).reshape(-1).shape[0])
            size = dat.stat().st_size // np.dtype(np.float32).itemsize
            if size % n:
                sys.exit(f"[{name}] {dat.name}: {size} float32 values are not "
                         f"divisible by n_params={n} from {tp.name}. Pass an "
                         f"explicit n_params in the .list.")
            return n
    sys.exit(f"[{name}] no *_truth.npy in {chains_dir}, so n_params cannot be "
             f"inferred and eval.py would skip every chain. Either run "
             f"make_truth_npy.py on this directory, or set n_params in the .list "
             f"(it will still be skipped without truths).")


def audit(chains_dir, n_params, name):
    """Warn about the two silent-drop failure modes before eval.py hits them."""
    dats = sorted(Path(chains_dir).glob("*.dat"))
    no_truth = [d.name for d in dats if not d.with_name(d.stem + "_truth.npy").exists()]
    n_logp = sum(1 for d in dats if d.with_name(d.stem + "_logp.npy").exists())
    if no_truth:
        print(f"  [warn] {len(no_truth)}/{len(dats)} chains have no _truth.npy "
              f"and will be SKIPPED (e.g. {no_truth[0]})", file=sys.stderr)
    if n_logp == 0:
        print(f"  [warn] no _logp.npy: --dlogp filtering is a NO-OP for this arm, "
              f"so it is compared UNFILTERED against filtered arms",
              file=sys.stderr)
    elif n_logp < len(dats):
        print(f"  [warn] only {n_logp}/{len(dats)} chains have _logp.npy; "
              f"filtering is applied inconsistently within this arm",
              file=sys.stderr)
    return len(dats) - len(no_truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="append", default=[], dest="lists",
                    required=True, metavar="FILE.list",
                    help="repeatable; arms are concatenated in the order given")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--thin", type=int, default=11)
    ap.add_argument("--dlogp", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--filtered-only", action="store_true")
    ap.add_argument("--no-intersect", action="store_true",
                    help="evaluate each arm on its own sims instead of the "
                         "shared intersection (GV/calibration stop being "
                         "comparable across arms -- diagnostic use only)")
    ap.add_argument("--full-table", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and audit the arms, then stop")
    ap.add_argument("--warm-cache", action="store_true",
                    help="only build each arm's thinned-chain cache (.npz next "
                         "to the chains), then stop -- run one job per list in "
                         "parallel before a big combined evaluation")

    ap.add_argument("--n-corner", type=int, default=1, help="Number of corner plots to generate")
    ap.add_argument("--no-seeds", action="store_true",
                    help="SBC/corner: do not draw the thin dashed line of each seed")
    ap.add_argument("--lw-scale", type=float, default=None,
                    help="SBC/corner: multiply the line widths (e.g. 0.7)")
    ap.add_argument("--walker-cut", type=float, default=None,
                    help="stuck-walker cut passed to eval.py (default 5; <=0 = off)")
    args = ap.parse_args()

    arms = []
    seen = {}
    for lf in args.lists:
        if not os.path.isfile(lf):
            sys.exit(f"list file not found: {lf}")
        print(f"\n=== {lf} ===")
        for name, p, npar, subs in parse_list(lf):
            if any(x == "*" for x in subs):
                dirs = list_chain_dirs(p, name)
                if not dirs:
                    sys.exit(f"[{name}] '*' matched no chains dir under {p}")
            elif subs:
                dirs = [resolve_chains_dir(p, x, name) for x in subs]
            else:
                dirs = [resolve_chains_dir(p, None, name)]

            tag = len(dirs) > 1
            for cdir in dirs:
                aname = f"{name} [{cdir.name}]" if tag else name
                n = npar or infer_n_params(cdir, aname)
                n_ok = audit(cdir, n, aname)
                if aname in seen:
                    sys.exit(f"duplicate arm name {aname!r} (in {seen[aname]} "
                             f"and {lf}); names key the legend and the metrics "
                             f"table, so they must be unique")
                seen[aname] = lf

                group_name = Path(lf).stem

                print(f"  {aname:34s} n_params={n}  usable={n_ok:4d}  {cdir}")
                arms.append({"name": aname, "dir": str(cdir), "n_params": n, "group": group_name})

    if not arms:
        sys.exit("no arms")

    os.makedirs(args.out, exist_ok=True)
    arms_path = os.path.join(args.out, "arms.json")
    json.dump(arms, open(arms_path, "w"), indent=2)
    print(f"\n{len(arms)} arms -> {arms_path}")

    if args.dry_run:
        print("--dry-run: stopping before eval.py")
        return

    here = os.path.dirname(os.path.abspath(__file__))

    if args.warm_cache:
        sys.path.insert(0, os.path.join(here, "sbi", "validation"))
        import eval as ev
        if args.walker_cut is not None:
            ev.WALKER_CUT = args.walker_cut if args.walker_cut > 0 else None
        for a in arms:
            ev.prefetch_dir(a["dir"], a["n_params"], args.burnin_frac, args.thin)
        print("--warm-cache: cache built, stopping before eval.py")
        return

    cmd = [sys.executable, "-u", os.path.join(here, "sbi", "validation", "eval.py"),
           "--config", arms_path, "--out", args.out,
           "--thin", str(args.thin), "--dlogp", str(args.dlogp),
           "--nbins", str(args.nbins), "--burnin-frac", str(args.burnin_frac),
           "--n-corner", str(args.n_corner)]

    if args.filtered_only:
        cmd.append("--filtered-only")
    if args.no_intersect:
        cmd.append("--no-intersect")
    if args.full_table:
        cmd.append("--full-table")
    if args.no_seeds:
        cmd.append("--no-seeds")
    if args.lw_scale is not None:
        cmd += ["--lw-scale", str(args.lw_scale)]
    if args.walker_cut is not None:
        cmd += ["--walker-cut", str(args.walker_cut)]
    subprocess.run(cmd, check=True)

    rep = json.load(open(os.path.join(args.out, "metrics.json")))
    for mode, rs in rep.items():
        got = {r["name"] for r in rs}
        lost = [a["name"] for a in arms if a["name"] not in got]
        print(f"[{mode}] {len(rs)}/{len(arms)} arms produced metrics")
        for n in lost:
            print(f"  DROPPED: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
