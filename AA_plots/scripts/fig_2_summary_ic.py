#!/usr/bin/env python3
"""Paper Fig. 2 (fig:summary_ic): I_c of each hand-crafted summary statistic, one A&A column."""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, mm layout, save
import paper_colors as pc  # noqa: E402        the paper colour scheme

INK = pc.INK
Z_COLORS = dict(pc.Z_COLORS)
Z_MARKERS = dict(pc.Z_MARKERS)
MARKER_EDGE = INK
REF_COLOR = INK
LEGEND_FRAME = INK
LEGEND_FACE = pc.WHITE

FS_L = {"label": aa.AX["label"], "tick": aa.AX["tick"], "legend": aa.AX["legend"] + 0.5,
        "family": aa.AX["panel"]}
IC_MS, IC_LW = 5.4, 1.3
LEGEND_IN = ("PDF", "PS")

CACHE = HERE.parents[1] / "paper_plots" / "cache" / "summary_information.npz"
IMAGES = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images"))
NAME = "fig_2_summary_ic"
Z_FILE = ["8.18", "10.32", "12.06"]
Z_PLOT = ["12.06", "10.32", "8.18"]
PDF_NAMES = [r"$\ell_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_5$", r"$\tau_6$"]
PS_NAMES = [rf"$k_{i}$" for i in range(1, 6)]
FAMILY = {"PDF": "PDF", "PS": "PS"}


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
    Ic = np.asarray(c["snr"])[[k[0] for k in cols]]
    return Ic, int(c["n_used"]), cols


def zstyle(z):
    col = Z_COLORS[z]
    return dict(color=col, marker=Z_MARKERS[z], ms=IC_MS, mfc=col, mec=MARKER_EDGE,
                mew=0.6, lw=IC_LW)


def plot(Ic, cols, name=NAME):
    W = aa.WIDTH_MM["column"]
    left, right, gap, bottom = 11.2, 0.8, 1.8, 6.3
    top, ax_h = 0.9, 35.4
    H = top + ax_h + bottom
    panels = ["PDF", "PS"]
    pw = (W - left - right - (len(panels) - 1) * gap) / len(panels)
    keep = [i for i, c in enumerate(cols) if c[2] != "mean"]
    lo = 0.5
    hi = 10.0 ** np.ceil(np.log10(1.5 * Ic[keep].max()))
    f1 = (0 - np.log10(lo)) / (np.log10(hi) - np.log10(lo))

    def legend(ax, b):
        hs = [Line2D([], [], **zstyle(z)) for z in Z_PLOT]
        kw = dict(loc="upper right", bbox_to_anchor=(0.975, 0.975))
        if b == "PS":
            kw = dict(loc="lower left", bbox_to_anchor=(0.025, f1 + 0.025))
        lg = ax.legend(hs, [rf"$z={z}$" for z in Z_PLOT], fontsize=FS_L["legend"] - 0.5,
                       frameon=True, fancybox=False, framealpha=1.0,
                       edgecolor=LEGEND_FRAME, facecolor=LEGEND_FACE,
                       borderpad=0.35, labelspacing=0.25, handlelength=2.0,
                       handletextpad=0.5, borderaxespad=0.0, **kw)
        lg.get_frame().set_linewidth(aa.LW["axes"])
        lg.set_zorder(5)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        x0 = left
        for p, b in enumerate(panels):
            ax = fig.add_axes([x0 / W, bottom / H, pw / W, ax_h / H])
            ax.axhline(1.0, color=REF_COLOR, lw=1.0, ls=aa.DASH, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            e0, e1 = int(np.ceil(np.log10(lo))), int(np.log10(hi))
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(e0, e1 + 1)]))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))
            ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", labelsize=FS_L["tick"], pad=aa.AX["tickpad"])
            if p == 0:
                ax.set_ylabel(r"$I_c$", fontsize=FS_L["label"], labelpad=aa.AX["labelpad"])
            else:
                ax.tick_params(axis="y", which="both", labelleft=False)

            for z in Z_PLOT:
                idx = [i for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                ax.plot(np.arange(5), Ic[idx], zorder=3, **zstyle(z))
            names = PDF_NAMES if b == "PDF" else PS_NAMES
            ax.set_xlim(-0.5, len(names) - 0.5)
            ax.set_xticks(range(len(names)), names, fontsize=FS_L["tick"])
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.tick_params(axis="x", which="major", length=0.9 * aa.TICK["major"], pad=aa.AX["tickpad"])

            ax.text(0.035, 0.965, rf"$\mathbf{{({'ab'[p]})}}$ {FAMILY[b]}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=FS_L["family"], color=INK, zorder=6)
            if b in LEGEND_IN:
                legend(ax, b)
            if p == len(panels) - 1:
                ax.text(0.97, f1 + 0.02, r"$I_c=1$", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=FS_L["tick"], color=INK, zorder=6)
            x0 += pw + gap

        aa.save(fig, name, outdir=IMAGES)
        plt.close(fig)


def main():
    aa.use()
    Ic, n_used, cols = load()
    print(f"I_c from {n_used} simulations")
    plot(Ic, cols)
    m = [v for v, c in zip(Ic, cols) if c[2] == "mean"]
    print("  <dTb> I_c (z = " + ", ".join(Z_PLOT) + "): " +
          ", ".join(f"{v:.1e}" for v in m) + "  -> caption")


if __name__ == "__main__":
    main()
