#!/usr/bin/env python3
"""Paper Fig. C1: Path B compressor ablation -- constraining power vs calibration, one A&A column."""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, mm layout, save
import paper_colors as pc  # noqa: E402        the paper colour scheme

IMAGES = HERE.parent / "images"
NAME = "fig_C1_ablation"

D = [("Full", 1.00, [1.63, 1.75, 1.02, 0.69], 14.95, "full"),
     ("J1: no SE", 1.05, [0.98, 1.65, 3.37, 1.19], 14.86, "comp"),
     ("J2: no shortcut", 1.41, [1.92, 1.18, 2.11, 0.86], 14.78, "comp"),
     ("J3: no SE, no shortcut", 1.46, [2.22, 1.70, 1.72, 1.62], 14.69, "comp"),
     ("J4: read-out block 5", 1.70, [1.32, 1.77, 1.31, 1.78], 14.72, "comp"),
     ("J6: no block 5", 1.13, [0.94, 1.47, 2.76, 1.10], 3.87, "small"),
     ("J7: half channels", 1.49, [1.56, 1.05, 1.46, 1.24], 3.89, "small")]
CHI_LIMIT = 1.47
J5_TEXT = ("J5: generic CNN, failed\n"
           r"($\mathrm{GV}\times3{\times}10^5$, mean $\hat\chi\approx25$)")

COL = dict(pc.ABLATION)
INK, MUTED, GUIDE = pc.INK, pc.MUTED, pc.GUIDE
MARKER = {"full": "*", "comp": "o", "small": "s"}
LEGEND = {"full": "full compressor", "comp": "component removed",
          "small": r"4$\times$ smaller network"}

FS = {"label": aa.AX["label"], "tick": aa.AX["tick"], "legend": aa.AX["legend"],
      "name": aa.FS["note"] + 0.5, "count": aa.FS["note"], "note": aa.FS["note"] - 0.4}
AREA_PER_M = 9.0
SHAPE_AREA = {"comp": 1.0, "small": np.pi / 4, "full": 1 / 0.48}
COUNT_DARKEN = {"full": 0.0, "comp": 0.25, "small": 0.5}
LEGEND_MS = {"full": 6.0, "comp": 5.0, "small": 4.4}
XLIM, YLIM = (0.92, 2.00), (1.17, 2.06)
W = aa.WIDTH_MM["column"]
H = 44.0
LEFT, TOP, BOTTOM = 10.5, 1.0, 8.6
BOX_W = W - 10.5 - 1.5
LEG_X = 3.0

LABEL_POS = {"Full": ((12, 0), "left"),
             "J1: no SE": ((10, 0), "left"),
             "J2: no shortcut": ((0, -15), "center"),
             "J3: no SE, no shortcut": ((0, 14), "center"),
             "J4: read-out block 5": ((-10, 14), "center"),
             "J6: no block 5": ((7, 0), "left"),
             "J7: half channels": ((7, 0), "left")}


def plot():
    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        ax = fig.add_axes([LEFT / W, BOTTOM / H, BOX_W / W,
                           (H - TOP - BOTTOM) / H])
        ax.axhline(CHI_LIMIT, ls=aa.DASH, lw=0.7, color=GUIDE, zorder=0)
        ax.text(0.935, CHI_LIMIT + 0.005, rf"$\hat\chi={CHI_LIMIT}$", fontsize=FS["note"],
                color=INK, ha="left", va="bottom")

        for n, g, chi, p, grp in D:
            y = np.mean(chi)
            ax.scatter(g, y, s=AREA_PER_M * p * SHAPE_AREA[grp], marker=MARKER[grp],
                       color=COL[grp], edgecolor=INK, linewidth=aa.LW["edge"], zorder=3)
            (dx, dy), ha = LABEL_POS[n]
            ax.annotate(n, (g, y), xytext=(dx, dy + 3.2), textcoords="offset points",
                        fontsize=FS["name"], ha=ha, va="center", color=INK)
            ax.annotate(rf"{p:.2f}$\,$M", (g, y), xytext=(dx, dy - 3.8),
                        textcoords="offset points", fontsize=FS["count"], ha=ha,
                        va="center", color=aa.darken(COL[grp], COUNT_DARKEN[grp]), fontweight="bold")

        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        aa.minor_ticks(ax, x=0.05, y=0.05)
        ax.tick_params(labelsize=FS["tick"], pad=aa.AX["tickpad"])
        ax.set_xlabel(r"$\mathrm{GV}/\mathrm{GV}_{\rm full}$  (less constraining $\rightarrow$)",
                      fontsize=FS["label"], labelpad=aa.AX["labelpad"])
        ax.set_ylabel(r"mean $\hat\chi_j$  (worse $\rightarrow$)",
                      fontsize=FS["label"], labelpad=aa.AX["labelpad"])

        h = [Line2D([], [], marker=MARKER[k], ls="", color=COL[k], mec=INK,
                    mew=aa.LW["edge"], ms=LEGEND_MS[k], label=LEGEND[k]) for k in COL]
        ax.legend(handles=h, loc="upper right", fontsize=FS["legend"], frameon=False,
                  handletextpad=0.4, borderaxespad=0.6, labelspacing=0.45)
        aa.save(fig, NAME, formats=("pdf", "svg", "png"), outdir=IMAGES)
        plt.close(fig)


if __name__ == "__main__":
    aa.use()
    plot()
