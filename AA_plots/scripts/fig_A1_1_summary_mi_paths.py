#!/usr/bin/env python3
"""Paper Fig. A1.1: r_I of the learned summaries t_i, compact, one A&A column (90 mm)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402
import paper_colors as pc  # noqa: E402

CACHE_DIR = ROOT / "paper_plots" / "cache"
SCR = Path("/gscratch/ddehiwalage-don/sbi")
BLOCKS = [
    ("a", r"Path A, VMIM", CACHE_DIR / "patha_mi.npz",
     "configs_paper/patha/arm_mlp_vmim_t8.yaml", SCR / "mlp_paper/vmim_t8_s43/summaries"),
    ("b", r"Path B, VMIM", CACHE_DIR / "pathb_mi.npz",
     "cnn_paper_k8/vmim/n1_t8_L8h128_s43", SCR / "cnn_paper_k8/vmim/n1_t8_L8h128_s43/summaries"),
    ("c", r"Path B, VMIM", CACHE_DIR / "pathb_mi.npz",
     "configs_paper/pathb/vmim/arm_cnn_vmim_n1.yaml", SCR / "cnn_paper/vmim/n1_s43/summaries"),
    ("d", r"Path B, MSE", CACHE_DIR / "pathb_mi.npz",
     "configs_paper/pathb/mse/arm_cnn_mse_n1.yaml", SCR / "cnn_paper/mse/n1_s43/summaries"),
]
GROUPS = [(r"(a) VMIM, $k=8$", ["a", "b"], ["Path A", "Path B"]),
          (r"(b) Path B, $k=4$", ["c", "d"], ["VMIM", "MSE"])]
N_SIMS, K_NN, RNG_SEED = 9120, 5, 0
PARAMS = [r"$\log_{10}f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$", r"$\log_{10}M_{\rm min}$"]
IMAGES = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images"))
NAME = "fig_A1_1_summary_mi_paths"

INK = pc.INK
HM_CMAP = pc.SEQ_CMAP
HM_GRID, HM_FRAME = pc.WHITE, INK
HATCH_FACE, HATCH_COLOR, HATCH = pc.HATCH_FACE, pc.HATCH_COLOR, "////"
HATCH_RC = {"hatch.linewidth": 0.45, "hatch.color": HATCH_COLOR}
CB_LABEL = r"$r_I=\sqrt{1-e^{-2I}}$"
FLOOR_TEXT = r"$r_I\approx 0$ (below the estimator floor)"
W = aa.WIDTH_MM["column"]
x_fam, x_stat, hm_x = 3.4, 8.8, 9.8
CB_GAP, CB_W, RIGHT = 2.6, 1.8, 9.6
PGAP, RH, GGAP = 1.4, 4.0, 1.6
Y_PAR, Y_SUB0, Y_HM0 = 3.6, 7.6, 8.8
SUB_H = 3.4
FS_L = {"label": aa.AX["label"], "tick": aa.AX["tick"], "legend": aa.AX["legend"],
        "family": aa.AX["panel"]}
CELL_FS = 6.6
SUB_FS = FS_L["tick"] - 1.0


def r_of(mi):
    return np.sqrt(1 - np.exp(-2 * np.clip(mi, 0, None)))


def compute_mi(d):
    """(mi, mi_null), each (n_param, k), from one noise realisation per simulation."""
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(RNG_SEED)
    ids = np.load(d / "original_sim_ids.npy", mmap_mode="r")
    _, _, counts = np.unique(ids, return_index=True, return_counts=True)
    order = np.argsort(ids, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows = order[starts + rng.integers(0, counts)]
    rows = np.sort(rng.choice(rows, min(N_SIMS, len(rows)), replace=False))
    t = np.load(d / "t.npy", mmap_mode="r")[rows].astype(np.float64)
    th = np.load(d / "theta.npy", mmap_mode="r")[rows].astype(np.float64)[:, :len(PARAMS)]

    def mi_of(tt, th_):
        return np.array([mutual_info_regression(tt, th_[:, j], n_neighbors=K_NN,
                                                random_state=RNG_SEED)
                         for j in range(th_.shape[1])])
    return mi_of(t, th), mi_of(t, th[rng.permutation(len(th))])


def data():
    """{letter: r_I (param, k)}, floor = max shuffled r_I over the blocks."""
    R, null = {}, []
    for letter, _, cache, cid, d in BLOCKS:
        have = dict(np.load(cache)) if cache.exists() else {}
        if f"mi|{cid}" not in have:
            print(f"  computing r_I of {d}", flush=True)
            have[f"mi|{cid}"], have[f"null|{cid}"] = compute_mi(d)
            np.savez(cache, **have)
        R[letter] = np.round(r_of(have[f"mi|{cid}"][:len(PARAMS)]), 2)
        null.append(r_of(have[f"null|{cid}"][:len(PARAMS)]).max())
    return R, float(max(null))


def block(fig, H, x0, y0, w, h, M, floor):
    """One framed block, top-left at (x0, y0) mm, w x h mm; M (rows, cols) = r_I."""
    n, m = M.shape
    ax = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])
    ax.imshow(np.ma.masked_less(M, floor), aspect="auto", cmap=HM_CMAP, vmin=0, vmax=1,
              extent=(-0.5, m - 0.5, n - 0.5, -0.5), interpolation="nearest", zorder=1)
    ax.set_xlim(-0.5, m - 0.5); ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for k in range(m):
            v = M[i, k]
            if v < floor:
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, facecolor=HATCH_FACE,
                                       edgecolor=HATCH_COLOR, hatch=HATCH, lw=0, zorder=2))
                continue
            c = aa.text_on(HM_CMAP(float(v)), dark=INK, light=pc.WHITE)
            ax.text(k, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center",
                    fontsize=CELL_FS, color=c, zorder=3)
    for k in np.arange(0.5, m - 0.5):
        ax.axvline(k, color=HM_GRID, lw=0.7, zorder=2.5)
    for i in np.arange(0.5, n - 0.5):
        ax.axhline(i, color=HM_GRID, lw=0.7, zorder=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which="both", left=False, bottom=False, top=False, right=False)
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_linewidth(aa.LW["axes"]); sp.set_color(HM_FRAME); sp.set_zorder(4)


def plot(R, floor):
    npar = len(PARAMS)
    nsub = max(len(g[1]) for g in GROUPS)
    hm_w = W - hm_x - CB_GAP - CB_W - RIGHT
    cw = (hm_w - (npar - 1) * PGAP) / (npar * nsub)
    nrow = [R[g[1][0]].shape[1] for g in GROUPS]
    foot = 5.6 if floor > 0 else 1.2
    H = Y_HM0 + sum(nrow) * RH + (len(GROUPS) - 1) * (GGAP + SUB_H) + foot

    def txt(x, y, s_, **kw):
        kw.setdefault("color", INK)
        fig.text(x / W, 1 - y / H, s_, **kw)

    def sub_headers(y, labels):
        for p in range(npar):
            x0 = hm_x + p * (nsub * cw + PGAP)
            for q, lab in enumerate(labels):
                txt(x0 + (q + 0.5) * cw, y, lab, ha="center", va="baseline", fontsize=SUB_FS)

    with mpl.rc_context(aa.rc()), mpl.rc_context(HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        for p in range(npar):
            x0 = hm_x + p * (nsub * cw + PGAP)
            txt(x0 + nsub * cw / 2, Y_PAR, PARAMS[p], ha="center", va="baseline",
                fontsize=FS_L["label"])
        y = Y_HM0
        for g, (glab, blocks, subs) in enumerate(GROUPS):
            if g:
                y += GGAP + SUB_H
            sub_headers(y - (Y_HM0 - Y_SUB0), subs)
            k = nrow[g]
            for p in range(npar):
                M = np.stack([R[b][p, :k] for b in blocks], axis=1)
                block(fig, H, hm_x + p * (nsub * cw + PGAP), y, len(blocks) * cw, k * RH,
                      M, floor)
            for i in range(k):
                txt(x_stat, y + (i + 0.5) * RH, rf"$t_{{{i + 1}}}$", ha="right",
                    va="center_baseline", fontsize=FS_L["tick"])
            num, rest = glab.split(" ", 1)
            txt(x_fam, y + k * RH / 2, rf"$\mathbf{{{num}}}$ {rest}", rotation=90,
                ha="center", va="center", fontsize=FS_L["family"], weight="bold")
            y += k * RH
        y_end = y
        y_cb0 = Y_HM0
        hx = hm_x

        cax = fig.add_axes([(hx + hm_w + CB_GAP) / W, 1 - y_end / H, CB_W / W,
                            (y_end - y_cb0) / H])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=HM_CMAP)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "", "0.5", "", "1"])
        cax.minorticks_off()
        cax.tick_params(which="both", direction="in", left=True, right=True,
                        length=0.6 * aa.TICK["major"], width=aa.LW["axes"],
                        labelsize=FS_L["tick"], pad=aa.AX["tickpad"])
        if floor > 0:
            cax.axhspan(0, floor, facecolor=HATCH_FACE, edgecolor=HATCH_COLOR, hatch=HATCH, lw=0,
                        zorder=3)
        cb.outline.set_linewidth(aa.LW["axes"]); cb.outline.set_edgecolor(HM_FRAME)
        cb.set_label(CB_LABEL, fontsize=FS_L["label"], labelpad=aa.AX["labelpad"], color=INK)
        if floor > 0:
            fig.legend([Patch(facecolor=HATCH_FACE, hatch=HATCH, edgecolor=HATCH_COLOR, lw=0.5)],
                       [FLOOR_TEXT], loc="lower center",
                       bbox_to_anchor=((hx + hm_w / 2) / W, 0.3 / H), fontsize=FS_L["legend"],
                       frameon=False, handlelength=1.4, handleheight=0.9, borderaxespad=0.0)
        aa.save(fig, NAME, outdir=IMAGES)
        plt.close(fig)
    print(f"  {W:.0f} x {H:.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args()
    aa.use()
    R, null_floor = data()
    floor = a.floor if a.floor is not None else null_floor
    print(f"  r_I floor (hatched below): {floor:.2f}  -> caption")
    plot(R, floor)


if __name__ == "__main__":
    main()
