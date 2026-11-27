"""Stage 4: SBC calibration and constraining power of groups of runs."""
import csv
import json
import os
import sys

import numpy as np

from ..config import load_config, run_dir
from ..evaluate import chain_sim, eval_run, list_chains


def _resolve(path, base_dir):
    """Path as given if it exists (relative to the cwd), else relative to base_dir."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(base_dir, path)


def group_runs(g, base_dir, overrides=None):
    """[(run label, chain dir)] of a group: `chains:` dirs, or every seed pair of `config:`."""
    if "chains" in g:
        return [(os.path.basename(os.path.dirname(d.rstrip("/"))) + "/" + os.path.basename(d.rstrip("/")), d)
                for d in g["chains"]]
    cfg = load_config(_resolve(g["config"], base_dir), overrides)
    if cfg["compressor"]["type"] == "none":
        cs_list = [0]
    else:
        cs_list = g.get("cseeds", cfg["seeds"]["compressor"])
    ns_list = g.get("nseeds", cfg["seeds"]["nle"])
    return [(f"c{cs}_n{ns}", str(run_dir(cfg, cs) / f"chains_n{ns}")) for cs in cs_list for ns in ns_list]


def n_params_of(d):
    """Number of parameters stored with the first chain in d."""
    f = next(iter(list_chains(d)), None)
    if f is None:
        sys.exit(f"no chains in {d}")
    return int(np.load(f[:-4] + "_truth.npy").size)


def plot_sbc(res, names, nbins, out):
    """Seed-mean SBC rank histograms per parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = len(names)
    fig, axes = plt.subplots(1, P, figsize=(3.2 * P, 3), sharey=True)
    x = (np.arange(nbins) + 0.5) / nbins
    for label, runs in res.items():
        if not runs:
            continue
        h = np.mean([r["rank_hist"] for r in runs], axis=0)
        for j in range(P):
            axes[j].step(x, h[j], where="mid", label=label)
    for j, ax in enumerate(axes):
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_title(names[j])
        ax.set_xlabel("rank / N")
    axes[0].set_ylabel("density (seed mean)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def add_args(ap):
    ap.add_argument("--out", default=None, help="output directory (default: `out` of the eval file)")


def run(args):
    # args.set applies to the run configs named in the groups (e.g. out_root=...)
    ev = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    params, names = ev["params"], ev["param_names"]
    out = args.out or ev["out"]
    os.makedirs(out, exist_ok=True)

    groups = {g["label"]: group_runs(g, base_dir, args.set) for g in ev["groups"]}
    sims = set.intersection(*[{chain_sim(f) for f in list_chains(d)} - {None}
                              for runs in groups.values() for _, d in runs])
    print(f"{sum(map(len, groups.values()))} runs, {len(sims)} shared simulations", flush=True)

    res, rows = {}, []
    for label, runs in groups.items():
        res[label] = []
        for run_name, d in runs:
            r = eval_run(d, n_params_of(d), params, ev, sims)
            if r is None:
                print(f"  [skip] {label} {run_name}: no usable chains", flush=True)
                continue
            r.update(group=label, run=run_name, dir=d)
            res[label].append(r)
            rows.append([label, run_name, r["used"], f"{r['gv']:.6e}"] + [f"{s:.6e}" for s in r["sigma"]]
                        + [f"{c:.3f}" for c in r["calib_ratio"]])
            print(f"  {label:28s} {run_name:12s} N={r['used']} GV={r['gv']:.3e} "
                  f"calib={[round(c, 2) for c in r['calib_ratio']]}", flush=True)

    head = ["group", "run", "n_used", "GV"] + [f"sigma_{n}" for n in names] + [f"calib_{n}" for n in names]
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([head] + rows)

    # seed average: geometric mean for GV and sigma (spread = std of log10), arithmetic for calib
    ghead = ["group", "n_runs", "GV", "GV_std_log10"]
    for n in names:
        ghead += [f"sigma_{n}", f"sigma_{n}_std_log10"]
    for n in names:
        ghead += [f"calib_{n}", f"calib_{n}_std"]
    grows = []
    print("\n=== group means ===")
    for label, rs in res.items():
        if not rs:
            continue
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        lg = np.log10([r["gv"] for r in rs])
        row = [label, len(rs), f"{10 ** lg.mean():.6g}", f"{sd(lg):.3g}"]
        for j in range(len(params)):
            ls = np.log10([r["sigma"][j] for r in rs])
            row += [f"{10 ** ls.mean():.6g}", f"{sd(ls):.3g}"]
        cal = np.array([r["calib_ratio"] for r in rs])
        for j in range(len(params)):
            row += [f"{cal[:, j].mean():.4g}", f"{sd(cal[:, j]):.3g}"]
        grows.append(row)
        print(f"  {label:28s} runs={len(rs)} GV={10 ** lg.mean():.3e} "
              f"calib={np.round(cal.mean(0), 2).tolist()}")
    with open(os.path.join(out, "metrics_groups.csv"), "w", newline="") as f:
        csv.writer(f).writerows([ghead] + grows)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot_sbc(res, names, ev["nbins"], os.path.join(out, "sbc.png"))
    print(f"\n-> {out}/ metrics.csv metrics_groups.csv metrics.json sbc.png")
