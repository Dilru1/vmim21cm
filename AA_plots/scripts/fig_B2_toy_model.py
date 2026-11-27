#!/usr/bin/env python3
"""Paper Fig. B2 (app:sbc_toy): Gaussian toy model of the rank histogram, in the style of fig 7 (bias ..."""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter, FormatStrFormatter
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sbi" / "validation" / "toy_model"))
sys.path.insert(0, str(ROOT / "sbi" / "validation"))
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402
import paper_colors as pc  # noqa: E402

IMAGES = HERE.parent / "images"
NAME = "fig_B2_toy_model"
SEEDS = 9

LABELS = {"cal": "calibrated", "bias": r"bias $0.15\sigma$", "over": "12 % too narrow",
          "under": "15 % too wide", "both": r"bias $0.12\sigma$, 8 % narrow",
          "theta": r"varying bias ($0.3\sigma$)", "skew": "skewed"}
MS = {"o": 3.6, "s": 3.4, "D": 3.2, "^": 4.2, "v": 4.2, "p": 4.2}
EDGE, EDGE_W = pc.INK, 0.45
TRUE_SCALE, TRUE_MEW = 1.8, 0.8
LEVELS = (1.26, 1.47, 2.5)
REGION = [pc.REGIONS_SEQ[k] for k in ("calibrated", "marginal", "miscal", "strong")]
REG_LAB = [r"$\hat\chi<1.26$", r"$1.26<\hat\chi<1.47$", r"$1.47<\hat\chi<2.5$", r"$\hat\chi>2.5$"]
CONTOUR_COLOR, GUIDE_COLOR = "0.35", pc.GUIDE
B_DATA, YLIM = (-0.24, 0.24), (0.84, 1.24)
LEG_W = 27.3
FS = dict(tick=aa.AX["tick"], label=aa.AX["label"], leg=aa.AX["table"])
W, H = aa.WIDTH_MM["column"], 44.0
LEFT, TOP, BOTTOM = 10.5, 1.0, 8.6
BOX_W = W - 10.5 - 1.5
LEG_X = 0.8
ROW = 3.3


def chi_grid(N, B=30, n=301):
    b = np.linspace(-0.45, 0.45, n)
    s = np.exp(np.linspace(np.log(0.75), np.log(1.45), n))
    bb, ss = np.meshgrid(b, s)
    e = norm.ppf(np.linspace(0, 1, B + 1))
    p = np.diff(norm.cdf(ss[..., None] * e + bb[..., None]), axis=-1)
    chi = (B * (p * (1 - p)).sum(-1) + N * B * ((p - 1 / B) ** 2).sum(-1)) / (B - 1)
    return bb, ss, chi


def main():
    import calib_ratio_toy_pulls as toyp
    from _palette import TOY_LOOK as _LOOK
    TOY_LOOK = {k: (lab, pc.TOY_COLORS[k], mk) for k, (lab, _, mk) in _LOOK.items()}
    aa.use()
    est = {k: toyp.toy_bs(k, seeds=SEEDS) for k in toyp.toy.CASES}
    for k, e in est.items():
        m, sd = e.mean(0), e.std(0, ddof=1)
        print(f"  {k:>6s}  b = {m[0]:+.3f} +- {sd[0]:.3f}   s = {m[1]:.3f} +- {sd[1]:.3f}"
              f"   (true {toyp.TRUE_BS[k][0]:+.3f}, {toyp.TRUE_BS[k][1]:.3f})")
    bb, ss, chi = chi_grid(toyp.toy.N)

    ph = H - TOP - BOTTOM
    pw = BOX_W
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / H)
    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        ax = fig.add_axes([fx(LEFT), fy(TOP + ph), pw / W, ph / H])
        ax.contourf(bb, ss, chi, levels=[0, *LEVELS, np.inf], colors=REGION, zorder=0)
        ax.contour(bb, ss, chi, levels=LEVELS, colors=CONTOUR_COLOR, linewidths=0.45, zorder=1)
        ax.axvline(0, color=GUIDE_COLOR, lw=0.35, ls=aa.DASH, zorder=0.8)
        ax.plot([-1, B_DATA[1]], [1, 1], color=GUIDE_COLOR, lw=0.35, ls=aa.DASH,
                zorder=0.8)
        for i, (k, (_, c, mk)) in enumerate(TOY_LOOK.items()):
            e = est[k]
            m, sd = e.mean(0), e.std(0, ddof=1)
            ax.plot(*toyp.TRUE_BS[k], mk, ms=MS[mk] * TRUE_SCALE, mfc="none", mec=c,
                    mew=TRUE_MEW, zorder=4)
            ax.errorbar(m[0], m[1], xerr=sd[0], yerr=sd[1], fmt="none", ecolor=c,
                        elinewidth=0.6, capsize=1.1, capthick=0.6, zorder=5 + 0.1 * i)
            ax.plot(*m, mk, ms=MS[mk], mfc=c, mec=EDGE, mew=EDGE_W, zorder=6 + 0.1 * i)
        ax.set_yscale("log")
        span = B_DATA[1] - B_DATA[0]
        ax.set_xlim(B_DATA[0], B_DATA[1] + span * (LEG_W + LEG_X) / (pw - LEG_W - LEG_X))
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_locator(FixedLocator([0.9, 1.0, 1.1, 1.2]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.85, YLIM[1], 0.025)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.2, 0.21, 0.05)))
        ax.tick_params(labelsize=FS["tick"], pad=aa.AX["tickpad"])
        ax.set_xlabel(r"bias $b$ ($\sigma$)", fontsize=FS["label"], labelpad=aa.AX["labelpad"])
        ax.set_ylabel(r"width ratio $s$", fontsize=FS["label"], labelpad=aa.AX["labelpad"])

        xl = LEFT + pw - LEG_X - LEG_W
        rows = list(TOY_LOOK.items())
        n = len(rows) + 2 + 0.3 + len(REG_LAB) + 0.6
        dy = min(ROW, ph / n)
        y0 = TOP + 0.5 * (ph - n * dy)
        for r, (k, (_, c, mk)) in enumerate(rows):
            y = y0 + (r + 0.5) * dy
            fig.add_artist(Line2D([fx(xl), fx(xl + 5.0)], [fy(y)] * 2, color=c, lw=0.6))
            fig.add_artist(Line2D([fx(xl + 2.5)], [fy(y)], ls="none", marker=mk, ms=MS[mk],
                                  mfc=c, mec=EDGE, mew=EDGE_W))
            fig.text(fx(xl + 6.5), fy(y), LABELS[k], ha="left", va="center_baseline",
                     fontsize=FS["leg"], color=pc.INK)
        y = y0 + (len(rows) + 0.5) * dy
        fig.add_artist(Line2D([fx(xl + 2.5)], [fy(y)], ls="none", marker="o",
                              ms=MS["o"] * TRUE_SCALE, mfc="none", mec=pc.INK, mew=TRUE_MEW))
        fig.text(fx(xl + 6.5), fy(y), "true value", ha="left", va="center_baseline",
                 fontsize=FS["leg"], color=pc.INK)
        y += dy
        fig.add_artist(Line2D([fx(xl), fx(xl + 5.0)], [fy(y)] * 2, color=pc.MUTED, lw=0.6))
        fig.add_artist(Line2D([fx(xl + 2.5)], [fy(y)], ls="none", marker="o", ms=MS["o"],
                              mfc=pc.MUTED, mec=EDGE, mew=EDGE_W))
        fig.text(fx(xl + 6.5), fy(y), rf"mean $\pm$ std, {SEEDS} sets", ha="left",
                 va="center_baseline", fontsize=FS["leg"], color=pc.INK)
        y += 0.5 * dy
        for v, lab in enumerate(REG_LAB):
            y += dy
            fig.add_artist(Rectangle((fx(xl), fy(y + 1.1)), 5.0 / W, 2.2 / H, fc=REGION[v],
                                     ec=CONTOUR_COLOR, lw=0.45, transform=fig.transFigure))
            fig.text(fx(xl + 6.5), fy(y), lab, ha="left", va="center_baseline",
                     fontsize=FS["leg"], color=pc.INK)
        aa.save(fig, NAME, formats=("pdf", "svg", "png"), outdir=IMAGES)
        plt.close(fig)


if __name__ == "__main__":
    main()
