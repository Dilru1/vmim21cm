#!/usr/bin/env python3
"""r_I heatmap of the Path B learned summaries t_i, same look as summary_mi_column ..."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_summary_information as fsi  # noqa: E402   heatmap blocks, colours, styles
import fig_t_information as fti  # noqa: E402         summaries_dir(yaml), SEED
aa = fsi.aa
INK = fsi.INK

CACHE = fsi.CACHE_DIR / "pathb_mi.npz"
N_SIMS, K_NN, RNG_SEED = 9120, 5, 0

NOISE = [(100, "n1"), (50, "nsqrt2"), (25, "n2")]
BLOCKS = [("MSE", "mse"), ("VMIM", "vmim")]
YAML = "configs_paper/pathb/{m}/arm_cnn_{m}_{n}.yaml"
PARAM_KEYS = ["fx", "tau", "rhs", "mmin"]


def one_row_per_sim(d, rng):
    """t, theta at one random noise realisation per simulation."""
    ids = np.load(d / "original_sim_ids.npy", mmap_mode="r")
    _, first, counts = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(ids, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows = order[starts + rng.integers(0, counts)]
    rows = np.sort(rng.choice(rows, min(N_SIMS, len(rows)), replace=False))
    t = np.load(d / "t.npy", mmap_mode="r")[rows].astype(np.float64)
    th = np.load(d / "theta.npy", mmap_mode="r")[rows].astype(np.float64)
    return t, th


def mi_of(t, th, seed):
    """(n_param, k) kNN MI I(t_i; theta_j) in nats."""
    return np.array([mutual_info_regression(t, th[:, j], n_neighbors=K_NN, random_state=seed)
                     for j in range(th.shape[1])])


def compute(recompute=False):
    """{yaml: (mi, mi_null)} cached in paper_plots/cache/pathb_mi.npz."""
    have = dict(np.load(CACHE)) if CACHE.exists() and not recompute else {}
    for _, m in BLOCKS:
        for _, n in NOISE:
            y = YAML.format(m=m, n=n)
            if f"mi|{y}" in have:
                continue
            d = fti.summaries_dir(y)
            print("  reading", d, flush=True)
            rng = np.random.default_rng(RNG_SEED)
            t, th = one_row_per_sim(d, rng)
            have[f"mi|{y}"] = mi_of(t, th, RNG_SEED)
            have[f"null|{y}"] = mi_of(t, th[rng.permutation(len(th))], RNG_SEED)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(CACHE, **have)
    return have


def r_of(mi):
    return np.sqrt(1 - np.exp(-2 * np.clip(mi, 0, None)))


def plot(R, floor, name="pathb_mi_column"):
    """R[block] = (n_param, k, n_noise) r_I."""
    npar, k, nn = next(iter(R.values())).shape
    FS_L = fsi.FS_L
    W = fsi.COLUMN_MM
    x_fam, x_stat, hm_x = 2.4, 10.6, 11.6
    cb_gap, cb_w, right = 1.6, 1.8, 9.6
    pgap, rh, bgap = 1.4, 4.6, 1.6
    y_par, y_z, y_hm = 3.6, 7.6, 8.8
    foot = 5.6 if floor > 0 else 1.2
    hm_w = W - hm_x - cb_gap - cb_w - right
    cw = (hm_w - (npar - 1) * pgap) / (npar * nn)
    H = y_hm + len(R) * k * rh + (len(R) - 1) * bgap + foot
    names = [rf"$t_{i + 1}$" for i in range(k)]

    with mpl.rc_context(aa.rc()), mpl.rc_context(fsi.HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        y = y_hm
        for b, (lab, _) in enumerate(BLOCKS):
            for p in range(npar):
                fsi._hm_block(fig, W, H, hm_x + p * (nn * cw + pgap), y, nn * cw, k * rh,
                              R[lab][p], floor)
            for s, nm in enumerate(names):
                fig.text(x_stat / W, 1 - (y + (s + 0.5) * rh) / H, nm, ha="right",
                         va="center_baseline", color=INK, fontsize=FS_L["tick"])
            fig.text(x_fam / W, 1 - (y + k * rh / 2) / H,
                     rf"$\mathbf{{({'ab'[b]})}}$ {lab}", rotation=90, ha="center",
                     va="center", fontsize=FS_L["family"] - 0.5, weight="bold", color=INK)
            y += k * rh + bgap
        y_end = y - bgap

        for p in range(npar):
            x0 = hm_x + p * (nn * cw + pgap)
            fig.text((x0 + nn * cw / 2) / W, 1 - y_par / H, fsi.PARAM_LABELS[PARAM_KEYS[p]],
                     ha="center", va="baseline", fontsize=FS_L["tick"], color=INK)
            for j, (h, _) in enumerate(NOISE):
                fig.text((x0 + (j + 0.5) * cw) / W, 1 - y_z / H, f"{h:g}", ha="center",
                         va="baseline", fontsize=FS_L["tick"] - 2.0, color=INK)
        fig.text((hm_x - 0.6) / W, 1 - y_z / H, r"$t_{\rm obs}$/h", ha="right",
                 va="baseline", fontsize=FS_L["tick"] - 2.0, color=INK)

        fsi._hm_colorbar(fig, W, H, hm_x + hm_w + cb_gap, y_hm, y_end - y_hm, cb_w, floor)
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
    for lab, m in BLOCKS:
        ys = [YAML.format(m=m, n=n) for _, n in NOISE]
        R[lab] = np.round(np.stack([r_of(have[f"mi|{y}"]) for y in ys], -1), 2)
        null += [r_of(have[f"null|{y}"]).max() for y in ys]
        print(f"\n{lab}: r_I  rows t_i, columns {' | '.join(PARAM_KEYS)} x {[h for h, _ in NOISE]} h")
        for i in range(R[lab].shape[1]):
            print(f"  t{i + 1}  " + "   ".join(" ".join(f"{v:.2f}" for v in R[lab][p, i])
                                            for p in range(R[lab].shape[0])))
    floor = a.floor if a.floor is not None else float(max(null))
    print(f"\n  r_I floor (hatched below): {floor:.2f}  -> caption")
    print("  mi: %.0f x %.0f mm" % plot(R, floor))


if __name__ == "__main__":
    main()
