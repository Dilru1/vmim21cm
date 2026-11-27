#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

from _sbc import plot_sbc
from _corner import plot_corner
from _latent import plot_latent

PATTERN = re.compile(r"_sim(\d+)_(?:row|offset)(\d+)")
COMMON_PARAMS = [0, 1, 2, 3]
LABELS = [r"$\log_{10}(F_x)$", r"$\tau$", r"$r_{H/S}$", r"$\log_{10}(M_{min})$", r"$f_{esc}$"]

def list_chains(d):
    return sorted(glob.glob(os.path.join(d, "*.dat")))

def chain_sim(fname):
    m = PATTERN.search(os.path.basename(fname))
    return int(m.group(1)) if m else None

_RAW = {}

def _env_cut():
    v = os.environ.get("SBI_EVAL_WALKER_CUT", "5")
    try:
        v = float(v)
    except ValueError:
        return None
    return v if v > 0 else None
WALKER_CUT = _env_cut()
WALKER_REF_PCT = 90.0

def walker_mask(fpath, nrows):
    """Row mask (True = keep) from the stuck-walker cut, or None if not applicable."""
    if WALKER_CUT is None:
        return None
    mpath, lpath = fpath[:-4] + "_meta.npy", fpath[:-4] + "_logp.npy"
    if not (os.path.exists(mpath) and os.path.exists(lpath)):
        return None
    meta = np.load(mpath)
    W, S, B = int(meta[0]), int(meta[1]), int(meta[2])
    if W <= 0 or (S - B) * W != nrows:
        return None
    L = np.load(lpath)
    if L.shape[0] != nrows:
        return None
    L = L.reshape(S - B, W).astype(np.float64)
    L[~np.isfinite(L)] = -np.inf
    wm = np.median(L, axis=0)
    fin = wm[np.isfinite(wm)]
    if fin.size == 0:
        return None
    good = wm >= np.percentile(fin, WALKER_REF_PCT) - WALKER_CUT
    return np.tile(good, S - B)

def _sig(fpath):
    def st(p):
        try:
            r = os.stat(p)
            return r.st_size, r.st_mtime_ns
        except FileNotFoundError:
            return -1, -1
    d = st(fpath)
    return (d[0], d[1], st(fpath[:-4] + "_truth.npy")[1], st(fpath[:-4] + "_logp.npy")[1])

def _read_raw(fpath, n_params, burnin_frac, thin):
    truth_path = fpath[:-4] + "_truth.npy"
    if not os.path.exists(truth_path):
        return None
    truth = np.load(truth_path).astype(np.float64)

    size = os.path.getsize(fpath) // np.dtype(np.float32).itemsize
    if size == 0 or size % n_params != 0:
        return None
    nrows = size // n_params
    chain = np.fromfile(fpath, dtype=np.float32, count=size).reshape(nrows, n_params)
    lp_all = None
    logp_path = fpath[:-4] + "_logp.npy"
    if os.path.exists(logp_path):
        lp_all = np.load(logp_path)
        if lp_all.shape[0] != nrows:
            lp_all = None
    wmask = walker_mask(fpath, nrows)
    if wmask is not None:
        chain = chain[wmask]
        lp_all = lp_all[wmask] if lp_all is not None else None
        nrows = chain.shape[0]

    b = int(burnin_frac * nrows)
    samples = np.ascontiguousarray(chain[b::thin])
    if samples.shape[0] < 20:
        return None

    logp = None
    if lp_all is not None:
        logp = np.ascontiguousarray(lp_all[b::thin])
    return samples, truth, logp

def _cache_path(d, n_params, burnin_frac, thin):
    wc = "" if WALKER_CUT is None else f"_w{WALKER_CUT:g}"
    return os.path.join(d, f".eval_cache_np{n_params}_b{burnin_frac:g}_t{thin}{wc}.npz")

def _load_npz(path):
    try:
        z = np.load(path)
        names, sigs, ok = z["names"], z["sigs"], z["ok"]
        so, to, lo, has_lp = z["s_off"], z["t_off"], z["l_off"], z["has_logp"]
        S, T, L = z["samples"], z["truth"], z["logp"]
    except Exception:
        return {}, {}
    ents, sg = {}, {}
    for i, n in enumerate(names.tolist()):
        sg[n] = tuple(int(x) for x in sigs[i])
        if not ok[i]:
            ents[n] = None
            continue
        ents[n] = (S[so[i]:so[i + 1]], T[to[i]:to[i + 1]],
                   L[lo[i]:lo[i + 1]] if has_lp[i] else None)
    return ents, sg

def _save_npz(path, ents, sg):
    names = sorted(ents)
    s_off, t_off, l_off = [0], [0], [0]
    S, T, L = [], [], []
    for n in names:
        e = ents[n]
        if e is not None:
            S.append(e[0]); T.append(e[1].reshape(-1))
            if e[2] is not None:
                L.append(e[2])
        s_off.append(s_off[-1] + (len(e[0]) if e else 0))
        t_off.append(t_off[-1] + (e[1].size if e else 0))
        l_off.append(l_off[-1] + (len(e[2]) if e and e[2] is not None else 0))
    ncol = S[0].shape[1] if S else 1
    tmp = path + f".tmp{os.getpid()}.npz"
    try:
        np.savez(tmp, names=np.array(names), sigs=np.array([sg[n] for n in names], np.int64).reshape(-1, 4),
                 ok=np.array([ents[n] is not None for n in names]),
                 has_logp=np.array([ents[n] is not None and ents[n][2] is not None for n in names]),
                 s_off=np.array(s_off, np.int64), t_off=np.array(t_off, np.int64), l_off=np.array(l_off, np.int64),
                 samples=np.concatenate(S) if S else np.zeros((0, ncol), np.float32),
                 truth=np.concatenate(T) if T else np.zeros(0),
                 logp=np.concatenate(L) if L else np.zeros(0))
        os.replace(tmp, path)
    except OSError as e:
        print(f"  [cache] could not write {path}: {e}", flush=True)
        try: os.remove(tmp)
        except OSError: pass

def prefetch_dir(d, n_params, burnin_frac, thin):
    """Fill the in-memory cache for every chain in `d`: reuse the .npz where the files are unchanged, read ..."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    key = (os.path.abspath(d), n_params, float(burnin_frac), int(thin))
    if key in _RAW:
        return
    t0 = time.time()
    files = list_chains(d)
    nw = int(os.environ.get("SBI_EVAL_WORKERS", "16"))
    use_npz = os.environ.get("SBI_EVAL_NOCACHE", "") != "1"
    cpath = _cache_path(d, n_params, burnin_frac, thin)
    old, old_sg = _load_npz(cpath) if use_npz and os.path.exists(cpath) else ({}, {})

    with ThreadPoolExecutor(nw) as ex:
        sigs = list(ex.map(_sig, files))
        ents, sg, todo = {}, {}, []
        for f, s in zip(files, sigs):
            n = os.path.basename(f)
            sg[n] = s
            if n in old and old_sg.get(n) == s:
                ents[n] = old[n]
            else:
                todo.append(f)
        for f, e in zip(todo, ex.map(lambda f: _read_raw(f, n_params, burnin_frac, thin), todo)):
            ents[os.path.basename(f)] = e
    _RAW[key] = ents
    print(f"  [cache] {d}: {len(files) - len(todo)} cached, {len(todo)} read "
          f"({nw} threads) in {time.time() - t0:.1f}s", flush=True)
    if use_npz and (todo or set(old) != set(ents)):
        _save_npz(cpath, ents, sg)

def load_chain(fpath, n_params, burnin_frac, thin, dlogp):
    key = (os.path.abspath(os.path.dirname(fpath)), n_params, float(burnin_frac), int(thin))
    name = os.path.basename(fpath)
    if key in _RAW and name in _RAW[key]:
        raw = _RAW[key][name]
    else:
        raw = _read_raw(fpath, n_params, burnin_frac, thin)
    if raw is None:
        return None
    samples, truth, logp = raw
    samples = samples.astype(np.float64)
    truth = truth.astype(np.float64)

    if dlogp is not None and logp is not None:
        logp = logp.astype(np.float64)
        if np.any(np.isfinite(logp)):
            keep = logp > (np.max(logp[np.isfinite(logp)]) - dlogp)
            if keep.sum() >= 20:
                samples = samples[keep]
    return samples, truth

def iqr_sigma(col):
    a = np.quantile(col, [0.15865, 1.0 - 0.15865])
    return 0.5 * (a[1] - a[0])

def sbc_rank(samples, truth, j, nbins):
    frac = np.mean(samples[:, j] < truth[j])
    return min(int(frac * nbins), nbins - 1)

def eval_arm(arm, burnin_frac, thin, dlogp, nbins, sims_keep):
    n_params = arm["n_params"]
    files = list_chains(arm["dir"])
    prefetch_dir(arm["dir"], n_params, burnin_frac, thin)
    rank_hist = np.zeros((len(COMMON_PARAMS), nbins))
    sigmas = [[] for _ in COMMON_PARAMS]
    sqrtdets = []
    scatter = {j: {"truth": [], "mean": [], "std": []} for j in range(len(COMMON_PARAMS))}
    used = 0
    for f in files:
        sim = chain_sim(f)
        if sim is None: continue
        if sims_keep is not None and sim not in sims_keep: continue
        loaded = load_chain(f, n_params, burnin_frac, thin, dlogp)
        if loaded is None: continue
        samples, truth = loaded
        sub = samples[:, COMMON_PARAMS]
        cov = np.cov(sub, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0 and np.isfinite(logdet):
            sqrtdets.append(np.exp(0.5 * logdet))
        for jj, j in enumerate(COMMON_PARAMS):
            sigmas[jj].append(iqr_sigma(samples[:, j]))
            rank_hist[jj, sbc_rank(samples, truth, j, nbins)] += 1.0
            scatter[jj]["truth"].append(truth[j])
            scatter[jj]["mean"].append(np.median(samples[:, j]))
            scatter[jj]["std"].append(iqr_sigma(samples[:, j]))
        used += 1
    if used == 0: return None
    rank_hist *= nbins / used
    calib = np.mean((rank_hist - 1.0) ** 2, axis=1)
    floor = (nbins - 1) / used
    calib_ratio = calib / floor
    tol_1sigma = math.sqrt(2.0 / (nbins - 1))
    calib_nsigma = (calib_ratio - 1.0) / tol_1sigma
    sqrtdets = np.asarray(sqrtdets)
    return {
        "name": arm["name"], "used": used,
        "group": arm.get("group", "default"),
        "dir": arm["dir"], "n_params": n_params,
        "sigma": [float(np.median(s)) for s in sigmas],
        "sigma_mean": [float(np.mean(s)) for s in sigmas],
        "gv": float(np.median(sqrtdets)),
        "gv_mean": float(np.mean(sqrtdets)),
        "calib": calib.tolist(),
        "calib_floor": float(floor),
        "calib_ratio": calib_ratio.tolist(),
        "calib_nsigma": calib_nsigma.tolist(),
        "rank_hist": rank_hist.tolist(),
        "scatter": {j: {k: list(map(float, v)) for k, v in scatter[j].items()} for j in scatter},
    }

def common_sims(arms):
    sets = []
    for a in arms:
        s = {chain_sim(f) for f in list_chains(a["dir"])}
        s.discard(None)
        sets.append(s)
    return set.intersection(*sets) if sets else set()

def _write_table(report, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "arm", "n_used", "GV_4x4", "GV_4x4_mean"]
                   + [f"sigma_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calibratio_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calibnsig_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calib_{LABELS[j]}" for j in COMMON_PARAMS])
        for mode, arms in report.items():
            for r in arms:
                w.writerow([mode, r["name"], r["used"],
                            f"{r['gv']:.6e}", f"{r['gv_mean']:.6e}"]
                           + [f"{s:.6e}" for s in r["sigma"]]
                           + [f"{c:.3f}" for c in r["calib_ratio"]]
                           + [f"{c:.2f}" for c in r["calib_nsigma"]]
                           + [f"{c:.4f}" for c in r["calib"]])
    print("\n=== GV (median sqrt|Sigma_4x4|) / sigma (median IQR) / calibration ===")
    for mode, arms in report.items():
        print(f"\n[{mode}]")
        for r in arms:
            print(f"  {r['name']:24s} N={r['used']:4d} GV={r['gv']:.4e} "
                  f"sigma={[round(s, 4) for s in r['sigma']]} "
                  f"calib_ratio={[round(c, 2) for c in r['calib_ratio']]} "
                  f"(1.0=perfect; floor={(r['calib_floor']):.3f})")

def _write_compact(report, path):
    npar = len(COMMON_PARAMS)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "arm", "n_used", "GV_median"]
                   + [f"sigma_{i + 1}" for i in range(npar)]
                   + [f"CR_{i + 1}" for i in range(npar)])
        for mode, arms in report.items():
            for r in arms:
                w.writerow([mode, r["name"], r["used"], f"{r['gv']:.6e}"]
                           + [f"{s:.6e}" for s in r["sigma"]]
                           + [f"{c:.3f}" for c in r["calib_ratio"]])


def _fmt_geo(m, s_log):
    if not np.isfinite(m) or m <= 0:
        return "nan"
    e = int(np.floor(np.log10(m)))
    sc = 10.0 ** e
    lo, hi = m - m * 10.0 ** (-s_log), m * 10.0 ** s_log - m
    return f"{m / sc:.2f}e{e:+03d} (-{lo / sc:.2f}e{e:+03d}/+{hi / sc:.2f}e{e:+03d})"


def _plot_scatter(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode, arms in report.items():
        cmap = plt.get_cmap("viridis"); m = max(len(arms) - 1, 1)
        fig, axes = plt.subplots(1, len(COMMON_PARAMS), figsize=(4 * len(COMMON_PARAMS), 4))
        for jj, j in enumerate(COMMON_PARAMS):
            ax = axes[jj]
            lims = []
            for i, r in enumerate(arms):
                s = r["scatter"].get(jj, r["scatter"].get(str(jj)))
                t, md = np.array(s["truth"]), np.array(s["mean"])
                ax.scatter(t, md, s=8, alpha=0.4, color=cmap(0.12 + 0.76 * i / m),
                           label=(r["name"] if jj == 0 else None))
                lims += [t.min(), t.max(), md.min(), md.max()]
            lo, hi = min(lims), max(lims)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(LABELS[j]); ax.set_xlabel("truth"); ax.set_ylabel("posterior median")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"[{mode}]"); fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", mode)
        fig.savefig(os.path.join(out, f"scatter_overlay_{safe}.png"), dpi=150)
        plt.close(fig)

def _write_seed_avg(report, path):
    rows = []
    for mode, arms in report.items():
        groups = {}
        for r in arms:
            base = r.get("group", "default")
            groups.setdefault(base, []).append(r)

        for base, g in groups.items():
            n = len(g)

            def geo_ms(vals):
                v = np.asarray(vals, float)
                lg = np.log10(np.clip(v, 1e-300, None))
                return float(10.0 ** lg.mean()), float(lg.std(ddof=1) if n > 1 else 0.0)

            def arith_ms(vals):
                v = np.asarray(vals, float)
                return float(v.mean()), float(v.std(ddof=1) if n > 1 else 0.0)

            gv_m, gv_s = geo_ms([r["gv"] for r in g])
            row = {"mode": mode, "arm": base, "n_seeds": n, "GV_mean": gv_m, "GV_sem": gv_s}

            for jj, j in enumerate(COMMON_PARAMS):
                sm, ss = geo_ms([r["sigma"][jj] for r in g])
                cm, cs = arith_ms([r["calib_ratio"][jj] for r in g])
                row[f"sigma_{LABELS[j]}_mean"] = sm; row[f"sigma_{LABELS[j]}_sem"] = ss
                row[f"calibratio_{LABELS[j]}_mean"] = cm; row[f"calibratio_{LABELS[j]}_sem"] = cs
            rows.append(row)

    if not rows: return
    full = path[:-4] + "_full.csv"
    keys = list(rows[0].keys())
    with open(full, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    npar = len(COMMON_PARAMS)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "arm", "n_seeds", "GV"]
                   + [f"sigma_{i + 1}" for i in range(npar)]
                   + [f"CR_{i + 1}" for i in range(npar)])
        for r in rows:
            sig = [_fmt_geo(r[f"sigma_{LABELS[j]}_mean"], r[f"sigma_{LABELS[j]}_sem"]) for j in COMMON_PARAMS]
            cr = [f"{r[f'calibratio_{LABELS[j]}_mean']:.2f} (±{r[f'calibratio_{LABELS[j]}_sem']:.2f})"
                  for j in COMMON_PARAMS]
            w.writerow([r["mode"], r["arm"], r["n_seeds"], _fmt_geo(r["GV_mean"], r["GV_sem"])] + sig + cr)
    print(f"[seed-avg] wrote {path} + {os.path.basename(full)} ({len(rows)} groups)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of arms: name, dir, n_params")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--thin", type=int, default=11)
    ap.add_argument("--dlogp", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--no-intersect", action="store_true")
    ap.add_argument("--filtered-only", action="store_true", help="skip unfiltered")
    ap.add_argument("--full-table", action="store_true", help="keep GV_mean column")
    ap.add_argument("--n-corner", type=int, default=1)
    ap.add_argument("--no-seeds", action="store_true",
                    help="SBC/corner: do not draw the thin dashed line of each seed")
    ap.add_argument("--lw-scale", type=float, default=None,
                    help="SBC/corner: multiply the line widths (e.g. 0.7)")
    ap.add_argument("--walker-cut", type=float, default=None,
                    help="drop MCMC walkers whose median logp is more than this below the "
                         "90th pct of walker medians (default 5 / env SBI_EVAL_WALKER_CUT; <=0 = off)")
    args = ap.parse_args()
    global WALKER_CUT
    if args.walker_cut is not None:
        WALKER_CUT = args.walker_cut if args.walker_cut > 0 else None
    print(f"[eval] stuck-walker cut: {WALKER_CUT}", flush=True)

    arms = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)

    sims_keep = None
    if not args.no_intersect:
        sims_keep = common_sims(arms)
        print(f"Shared sims across all arms: {len(sims_keep)}")
        if len(sims_keep) == 0:
            print("WARNING: no shared sims; falling back to per-arm sims.")
            sims_keep = None

    if args.filtered_only:
        modes = {f"filtered_dlogp{args.dlogp:g}": args.dlogp}
    else:
        modes = {"unfiltered": None, f"filtered_dlogp{args.dlogp:g}": args.dlogp}

    report = {}
    for mode, dlogp in modes.items():
        report[mode] = []
        for arm in arms:
            r = eval_arm(arm, args.burnin_frac, args.thin, dlogp, args.nbins, sims_keep)
            if r is not None: report[mode].append(r)

    json.dump(report, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    _write_table(report, os.path.join(args.out, "metrics_full.csv"))
    _write_compact(report, os.path.join(args.out, "metrics.csv"))

    try:
        from latex_table import write_latex
        write_latex(report, os.path.join(args.out, "metrics.tex"), compact=not args.full_table)
    except Exception as e:
        pass

    _write_seed_avg(report, os.path.join(args.out, "metrics_seed_avg.csv"))
    _plot_scatter(report, args.out)

    import _settings
    if args.no_seeds:
        _settings.SHOW_SEEDS = False
    if args.lw_scale is not None:
        _settings.LW_SCALE = args.lw_scale
    plot_sbc(report, args.out, args.nbins, COMMON_PARAMS, LABELS)

    _corner_dlogp = args.dlogp if args.filtered_only else None
    plot_corner(report, args.out, args.burnin_frac, args.thin, _corner_dlogp, args.n_corner, COMMON_PARAMS, LABELS)


    print(f"\nWrote {args.out}/: metrics.csv, metrics_full.csv, metrics_seed_avg.csv, "
          f"metrics_seed_avg_full.csv, metrics.tex, metrics.json, "
          f"scatter_overlay_*.png, sbc_overlay_*.png, corner_sim*_*.png")

if __name__ == "__main__":
    import sys
    sys.modules.setdefault("eval", sys.modules[__name__])
    main()
