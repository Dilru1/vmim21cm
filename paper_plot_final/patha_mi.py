#!/usr/bin/env python3
"""r_I heatmap of the Path A learned summaries t_i (VMIM, k = 4, 8, 16), same look as ..."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_summary_information as fsi  # noqa: E402   heatmap blocks, colours, styles
import fig_t_information as fti  # noqa: E402         summaries_dir(yaml), SEED
import pathb_mi as pbm  # noqa: E402                  one_row_per_sim, mi_of, r_of
aa = fsi.aa
INK = fsi.INK

CACHE = fsi.CACHE_DIR / "patha_mi.npz"
RNG_SEED = pbm.RNG_SEED

KS = [4, 8, 16]
YAML = "configs_paper/patha/arm_mlp_vmim_t{k}.yaml"
PARAM_KEYS = ["fx", "tau", "rhs", "mmin"]
VALUE_FS = 5.6


def compute(recompute=False):
    """{"mi|yaml", "null|yaml"} cached in paper_plots/cache/patha_mi.npz."""
    have = dict(np.load(CACHE)) if CACHE.exists() and not recompute else {}
    for k in KS:
        y = YAML.format(k=k)
        if f"mi|{y}" in have:
            continue
        d = fti.summaries_dir(y)
        print("  reading", d, flush=True)
        rng = np.random.default_rng(RNG_SEED)
        t, th = pbm.one_row_per_sim(d, rng)
        have[f"mi|{y}"] = pbm.mi_of(t, th, RNG_SEED)
        have[f"null|{y}"] = pbm.mi_of(t, th[rng.permutation(len(th))], RNG_SEED)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **have)
    return have


def plot(R, floor, name="patha_mi_full"):
    """R[k] = (n_param, k) r_I."""
    npar = len(PARAM_KEYS)
    FS_L = fsi.FS_L
    W = aa.WIDTH_MM["full"]
    x_par, hm_x = 16.0, 17.0
    cb_gap, cb_w, right, kgap = 2.0, 1.8, 10.0, 2.4
    rh = 5.0
    y_fam, y_t, y_hm = 3.8, 7.6, 8.8
    foot = 5.6 if floor > 0 else 1.2
    hm_w = W - hm_x - cb_gap - cb_w - right
    cw = (hm_w - (len(KS) - 1) * kgap) / sum(KS)
    H = y_hm + npar * rh + foot

    with mpl.rc_context(aa.rc()), mpl.rc_context(fsi.HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        x0 = hm_x
        for b, k in enumerate(KS):
            fsi._hm_block(fig, W, H, x0, y_hm, k * cw, npar * rh, R[k], floor, fs=VALUE_FS)
            fig.text((x0 + k * cw / 2) / W, 1 - y_fam / H,
                     rf"$\mathbf{{({'abc'[b]})}}$ $\boldsymbol{{k={k}}}$", ha="center",
                     va="baseline", fontsize=FS_L["family"] - 0.5, weight="bold", color=INK)
            for i in range(k):
                fig.text((x0 + (i + 0.5) * cw) / W, 1 - y_t / H, f"{i + 1}", ha="center",
                         va="baseline", fontsize=FS_L["tick"] - 2.0, color=INK)
            x0 += k * cw + kgap
        fig.text((hm_x - 0.6) / W, 1 - y_t / H, r"$t_i$, $i=$", ha="right", va="baseline",
                 fontsize=FS_L["tick"] - 2.0, color=INK)
        for j, p in enumerate(PARAM_KEYS):
            fig.text(x_par / W, 1 - (y_hm + (j + 0.5) * rh) / H, fsi.PARAM_LABELS[p],
                     ha="right", va="center_baseline", fontsize=FS_L["tick"], color=INK)

        fsi._hm_colorbar(fig, W, H, hm_x + hm_w + cb_gap, y_hm, npar * rh, cb_w, floor)
        if floor > 0:
            fsi._floor_legend(fig, W, H, hm_x + hm_w / 2)
        fsi.save(fig, name)
        plt.close(fig)
    return W, H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--floor", type=float, default=None,
                    help="r_I floor (hatched below). Default: max r_I with shuffled theta")
    a = ap.parse_args()
    aa.use()
    have = compute(a.recompute)

    R, null = {}, []
    for k in KS:
        y = YAML.format(k=k)
        R[k] = np.round(pbm.r_of(have[f"mi|{y}"]), 2)
        null.append(pbm.r_of(have[f"null|{y}"]).max())
        print(f"\nk = {k}: r_I  rows {PARAM_KEYS}, columns t_1..t_{k}")
        for j, p in enumerate(PARAM_KEYS):
            print(f"  {p:5s} " + " ".join(f"{v:.2f}" for v in R[k][j]))
    floor = a.floor if a.floor is not None else float(max(null))
    print(f"\n  r_I floor (hatched below): {floor:.2f}  -> caption")
    print("  mi: %.0f x %.0f mm" % plot(R, floor))


if __name__ == "__main__":
    main()
