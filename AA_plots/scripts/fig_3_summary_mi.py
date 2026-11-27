#!/usr/bin/env python3
"""Paper Fig. 3 (fig:summary_mi): r_I of each hand-crafted summary statistic, one A&A column."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, mm layout, save
import paper_colors as pc  # noqa: E402        the paper colour scheme

INK = pc.INK
HM_CMAP = pc.SEQ_CMAP
HM_GRID_COLOR = pc.WHITE
HM_FRAME_COLOR = INK
HM_TEXT_DARK = INK
HM_TEXT_LIGHT = pc.WHITE
HATCH_FACE = pc.HATCH_FACE
HATCH_COLOR = pc.HATCH_COLOR

FS_L = {"label": aa.AX["label"], "tick": aa.AX["tick"], "legend": aa.AX["legend"],
        "family": aa.AX["panel"]}
CELL_FS = 6.6
HATCH = "////"
HATCH_RC = {"hatch.linewidth": 0.45, "hatch.color": HATCH_COLOR}
CB_LABEL = r"$r_I=\sqrt{1-e^{-2I}}$"
FLOOR_TEXT = r"$r_I\approx 0$ (below the estimator floor)"

CACHE_DIR = HERE.parents[1] / "paper_plots" / "cache"
CACHE = CACHE_DIR / "summary_information.npz"
VOX_MI_CACHE = CACHE_DIR / "vox_mi.npz"
IMAGES = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images"))
NAME = "fig_3_summary_mi"
Z_FILE = ["8.18", "10.32", "12.06"]
Z_PLOT = ["12.06", "10.32", "8.18"]
PDF_NAMES = [r"$\ell_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_5$", r"$\tau_6$"]
PS_NAMES = [rf"$k_{i}$" for i in range(1, 6)]
FAMILY = {"PDF": "PDF", "PS": "PS", "mean": r"$\langle\delta T_b\rangle$"}
PARAMS = [r"$\log_{10} f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$",
          r"$\log_{10} M_{\rm min}$", r"$f_{\rm esc}$"]


def column_layout():
    """[(npz column, label, block, z)] in plotted order: PDF, PS, mean."""
    cols = []
    for b, names, off in (("PDF", PDF_NAMES, 0), ("PS", PS_NAMES, 15)):
        for zp in Z_PLOT:
            zi = Z_FILE.index(zp)
            cols += [(off + 5 * zi + i, names[i], b, zp) for i in range(5)]
    for zp in Z_PLOT:
        cols.append((30 + Z_FILE.index(zp), zp, "mean", zp))
    return cols


def load(cache=CACHE):
    c = np.load(cache)
    cols = column_layout()
    idx = [k[0] for k in cols]
    rI = np.sqrt(1.0 - np.exp(-2.0 * np.clip(c["mi"][:, idx], 0, None)))
    return rI, int(c["n_used"]), cols


def voxel_floor(cache=VOX_MI_CACHE):
    """Estimator floor: shuffled-parameter r_I of the single-voxel run (mean)."""
    c = np.load(cache)
    return float(np.mean([c[f"null{z}"] for z in Z_PLOT]))


def family_rI(rI, cols, j, b):
    """r_I of parameter j for every statistic of block b: (n_stat, n_z), plotted z order."""
    return np.array([[rI[j, i] for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                     for z in Z_PLOT]).T


def text_color(v):
    """Value text readable on the cell colour of r_I = v."""
    return aa.text_on(HM_CMAP(float(v)), dark=HM_TEXT_DARK, light=HM_TEXT_LIGHT)


def hm_block(fig, W, H, x0, y0, w, h, M, floor, fs=CELL_FS):
    """One framed block, top-left corner at (x0, y0) mm, size w x h mm."""
    n, m = M.shape
    ax = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])
    ax.imshow(np.ma.masked_less(M, floor), aspect="auto", cmap=HM_CMAP, vmin=0, vmax=1,
              extent=(-0.5, m - 0.5, n - 0.5, -0.5), interpolation="nearest", zorder=1)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for k in range(m):
            v = M[i, k]
            if v < floor:
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, facecolor=HATCH_FACE,
                                       hatch=HATCH, lw=0, zorder=2))
                continue
            ax.text(k, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center",
                    fontsize=fs, color=text_color(v), zorder=3)
    for k in np.arange(0.5, m - 0.5):
        ax.axvline(k, color=HM_GRID_COLOR, lw=0.7, zorder=2.5)
    for i in np.arange(0.5, n - 0.5):
        ax.axhline(i, color=HM_GRID_COLOR, lw=0.7, zorder=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which="both", left=False, bottom=False, top=False, right=False)
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_linewidth(aa.LW["axes"]); sp.set_color(HM_FRAME_COLOR); sp.set_zorder(4)
    return ax


def hm_colorbar(fig, W, H, x, y0, h, w, floor):
    """Framed vertical colour bar, top at y0 mm; label centred along it (rotated)."""
    cax = fig.add_axes([x / W, 1 - (y0 + h) / H, w / W, h / H])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=HM_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "", "0.5", "", "1"])
    cax.minorticks_off()
    cax.tick_params(which="both", direction="in", left=True, right=True,
                    length=0.6 * aa.TICK["major"], width=aa.LW["axes"],
                    labelsize=FS_L["tick"], pad=aa.AX["tickpad"])
    if floor > 0:
        cax.axhspan(0, floor, facecolor=HATCH_FACE, hatch=HATCH, lw=0, zorder=3)
    cb.outline.set_linewidth(aa.LW["axes"]); cb.outline.set_edgecolor(HM_FRAME_COLOR)
    cb.set_label(CB_LABEL, fontsize=FS_L["label"], labelpad=aa.AX["labelpad"], color=INK)
    return cb


def plot(rI, cols, floor, name=NAME):
    npar, nz = rI.shape[0], len(Z_PLOT)
    blocks = [("PDF", PDF_NAMES), ("PS", PS_NAMES), ("mean", [FAMILY["mean"]])]

    W = aa.WIDTH_MM["column"]
    x_fam, x_stat, hm_x = 3.4, 8.8, 9.8
    cb_gap, cb_w, right = 2.6, 1.8, 9.6
    pgap, rh, bgap = 1.4, 4.0, 1.6
    y_par, y_z, y_hm = 3.6, 7.6, 8.8
    foot = 5.6 if floor > 0 else 1.2
    hm_w = W - hm_x - cb_gap - cb_w - right
    cw = (hm_w - (npar - 1) * pgap) / (npar * nz)
    nrow = sum(len(n) for _, n in blocks)
    H = y_hm + nrow * rh + (len(blocks) - 1) * bgap + foot
    letters = iter("abcdefgh")

    with mpl.rc_context(aa.rc()), mpl.rc_context(HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        y = y_hm
        for b, names in blocks:
            n = len(names)
            for p in range(npar):
                hm_block(fig, W, H, hm_x + p * (nz * cw + pgap), y, nz * cw, n * rh,
                         family_rI(rI, cols, p, b), floor)
            for s, lab in enumerate(names):
                fig.text(x_stat / W, 1 - (y + (s + 0.5) * rh) / H, lab, ha="right",
                         va="center_baseline", color=INK, fontsize=FS_L["tick"])
            if b != "mean":
                fig.text(x_fam / W, 1 - (y + n * rh / 2) / H,
                         rf"$\mathbf{{({next(letters)})}}$ {FAMILY[b]}", rotation=90,
                         ha="center", va="center", fontsize=FS_L["family"],
                         weight="bold", color=INK)
            y += n * rh + bgap
        y_end = y - bgap

        for p in range(npar):
            x0 = hm_x + p * (nz * cw + pgap)
            fig.text((x0 + nz * cw / 2) / W, 1 - y_par / H, PARAMS[p], ha="center",
                     va="baseline", fontsize=FS_L["label"], color=INK)
            for k, z in enumerate(Z_PLOT):
                fig.text((x0 + (k + 0.5) * cw) / W, 1 - y_z / H, f"{float(z):.1f}",
                         ha="center", va="baseline", fontsize=FS_L["tick"] - 2.0, color=INK)
        fig.text((hm_x - 0.6) / W, 1 - y_z / H, r"$z$", ha="right", va="baseline",
                 fontsize=FS_L["tick"] - 1.0, color=INK)

        hm_colorbar(fig, W, H, hm_x + hm_w + cb_gap, y_hm, y_end - y_hm, cb_w, floor)
        if floor > 0:
            fig.legend([Patch(facecolor=HATCH_FACE, hatch=HATCH, edgecolor=INK, lw=0.5)],
                       [FLOOR_TEXT], loc="lower center",
                       bbox_to_anchor=((hm_x + hm_w / 2) / W, 0.3 / H),
                       fontsize=FS_L["legend"], frameon=False, handlelength=1.4,
                       handleheight=0.9, borderaxespad=0.0)
        aa.save(fig, name, outdir=IMAGES)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=float, default=None,
                    help="r_I estimator floor (default: shuffled-parameter floor of the "
                         "single-voxel run, from vox_mi.npz)")
    a = ap.parse_args()
    aa.use()
    rI, n_used, cols = load()
    floor = voxel_floor() if a.floor is None else a.floor
    print(f"MI from {n_used} simulations, one noise realisation each")
    print(f"  r_I estimator floor (hatched below): {floor:.2f}  -> caption")
    plot(rI, cols, floor)


if __name__ == "__main__":
    main()
