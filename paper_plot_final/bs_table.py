#!/usr/bin/env python3
"""Toy-model (b, s) table for every run of one or more stage-4 output dirs, written next to their ..."""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sbi", "validation"))
import eval as ev  # noqa: E402

THIN, BURNIN, DLOGP, MODE = 11, 0.0, 10.0, "filtered_dlogp10"
PARAMS = ["Fx", "tau", "rhs", "Mmin"]


def bs_from_u(u):
    sd = u.std(ddof=1)
    return -u.mean() / sd, 1.0 / sd


def run_bs(arm):
    ev.prefetch_dir(arm["dir"], arm["n_params"], BURNIN, THIN)
    u, z = [], []
    for f in ev.list_chains(arm["dir"]):
        if ev.chain_sim(f) is None:
            continue
        L = ev.load_chain(f, arm["n_params"], BURNIN, THIN, DLOGP)
        if L is None:
            continue
        x, t = L[0][:, ev.COMMON_PARAMS], L[1][ev.COMMON_PARAMS]
        u.append(norm.ppf(((x < t).sum(0) + 0.5) / (len(x) + 1)))
        z.append((t - x.mean(0)) / x.std(0))
    u, z = np.array(u), np.array(z)
    return ([bs_from_u(u[:, k]) for k in range(u.shape[1])],
            [bs_from_u(z[:, k]) for k in range(z.shape[1])], len(u))


def do_dir(d):
    runs = json.load(open(os.path.join(d, "metrics.json")))[MODE]
    hdr = ["group", "run", "n_sims"] + [f"{q}_{p}" for p in PARAMS for q in ("b", "s")] \
        + [f"{q}_ms_{p}" for p in PARAMS for q in ("b", "s")] + ["s_mean"]
    rows = []
    for r in runs:
        bs, ms, n = run_bs(r)
        row = [r.get("group", r["name"]), r["name"], n] + [v for b in bs for v in b] \
            + [v for b in ms for v in b] + [np.mean([b[1] for b in bs])]
        rows.append(row)
        print(f"  {r['name']:40s} n={n}  s " + "/".join(f"{b[1]:.3f}" for b in bs)
              + "  b " + "/".join(f"{b[0]:+.3f}" for b in bs), flush=True)
    fmt = lambda v: v if isinstance(v, str) else (str(v) if isinstance(v, int) else f"{v:.4f}")
    with open(os.path.join(d, "bs_table.csv"), "w") as fh:
        fh.write(",".join(hdr) + "\n")
        fh.writelines(",".join(fmt(v) for v in row) + "\n" for row in rows)
    num = hdr[3:]
    with open(os.path.join(d, "bs_table_seed_avg.csv"), "w") as fh:
        fh.write(",".join(["group", "n_runs"] + [f"{c}_mean,{c}_std" for c in num]) + "\n")
        for g in dict.fromkeys(row[0] for row in rows):
            A = np.array([row[3:] for row in rows if row[0] == g], float)
            sd = A.std(0, ddof=1) if len(A) > 1 else np.zeros(A.shape[1])
            fh.write(",".join([g, str(len(A))] + [f"{m:.4f},{s:.4f}" for m, s in zip(A.mean(0), sd)]) + "\n")
    print("->", os.path.join(d, "bs_table.csv"), "+ bs_table_seed_avg.csv", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, flush=True)
        do_dir(os.path.abspath(d))
