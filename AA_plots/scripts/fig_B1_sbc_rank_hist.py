#!/usr/bin/env python3
"""Paper Fig. B1 (fig:sbc_shrt): normalised SBC rank histograms, one A&A column (90 mm)."""
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402
import paper_colors as pc  # noqa: E402
import arm_look as al  # noqa: E402

IMAGES = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images"))
NAME = "fig_B1_sbc_rank_hist"
KEYS = list(al.LINE_KEYS)
NBINS = 30
NPAR = 4
PLAB = [r"$\log_{10}f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$", r"$\log_{10}M_{\rm min}$"]

LW = 0.55
GUIDE_LW = 0.9
BAND = {68: pc.SBC_BAND_68,
        95: pc.SBC_BAND_95}
GUIDE_COLOR = pc.MUTED
BIAS_LS = (0, (4, 1.5, 1, 1.5))
UNDER_LS = (0, (1, 1.2))
FS = {"tick": aa.AX["tick"], "label": aa.AX["label"], "panel": aa.AX["panel"]}
W = aa.WIDTH_MM["column"]
LEFT, RIGHT, TOP = 11.0, 2.4, 1.0
PH, VGAP = 17.0, 1.3
XLAB = 7.5
Y_TICKS = (0.5, 1.0, 1.5)
LEG_COL_W = [1.28, 1.0, 0.95, 0.80]
LEG_GAP = 1.5


def reference_curves(n_used):
    """68 / 95 % bands, and the expected histograms for a 0.2 sigma bias and s = 0.85."""
    s = math.sqrt(NBINS * (1.0 - 1.0 / NBINS) / n_used)
    bands = (1 - s, 1 + s, max(1 - 1.96 * s, 0), 1 + 1.96 * s)
    std, u, w = NormalDist(), np.linspace(0.0, 1.0, 401), 1.0 / NBINS

    def expected(loc, scale):
        F = NormalDist(loc, scale).cdf
        out = []
        for a in np.clip(u - w / 2, 0, 1 - w):
            za = std.inv_cdf(a) if a > 0 else -math.inf
            zb = std.inv_cdf(a + w) if a + w < 1 else math.inf
            fa = 0.0 if za == -math.inf else F(za)
            fb = 1.0 if zb == math.inf else F(zb)
            out.append((fb - fa) / w)
        return u, np.array(out)
    return bands, expected(0.2, 1.0), expected(0.0, 0.85)


def load():
    """{key: mean rank histogram (NPAR, NBINS)}, max n_used."""
    H, n_used = {}, 0
    for k in KEYS:
        rs = al.runs(k)
        H[k] = np.mean([np.asarray(r["rank_hist"])[:NPAR] for r in rs], axis=0)
        n_used = max(n_used, max(r["used"] for r in rs))
        print(f"  {k:7s} {len(rs)} runs  ({al.SOURCES[k][1]})")
    return H, n_used


def nice_range(lo, hi):
    pad = 0.06 * (hi - lo)
    return math.floor((lo - pad) * 20) / 20, math.ceil((hi + pad) * 20) / 20


def plot(H, n_used):
    (l1, h1, l2, h2), bias, under = reference_curves(n_used)
    vals = np.concatenate([h.ravel() for h in H.values()])
    ylo, yhi = nice_range(min(l2, vals.min()), max(h2, vals.max()))
    edges = np.linspace(0, 1, NBINS + 1)

    def band(fig, W_, H_, x, y, alpha):
        fig.add_artist(Rectangle((x / W_, 1 - (y + 1.0) / H_), al.HANDLE_W / W_, 2.0 / H_,
                                 fc=BAND[alpha], ec="none", transform=fig.transFigure))

    def guide(ls):
        def draw(fig, W_, H_, x, y):
            fig.add_artist(Line2D([x / W_, (x + al.HANDLE_W) / W_], [1 - y / H_] * 2,
                                  color=GUIDE_COLOR, ls=ls, lw=GUIDE_LW, transform=fig.transFigure))
        return draw
    extra = [("SBC guides", [(lambda f, a, b, x, y: band(f, a, b, x, y, 68), "68 %"),
                             (lambda f, a, b, x, y: band(f, a, b, x, y, 95), "95 %"),
                             (guide(BIAS_LS), r"bias $0.2\sigma$"),
                             (guide(UNDER_LS), r"$s=1.15$")])]
    keys = [k for k in KEYS if k in H]
    cols = al.default_columns(keys, extra)
    leg_h = al.legend_height(cols)
    Hfig = TOP + NPAR * PH + (NPAR - 1) * VGAP + XLAB + LEG_GAP + leg_h
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / Hfig)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, Hfig * aa.MM))
        pw = W - LEFT - RIGHT
        for j in range(NPAR):
            y0 = TOP + j * (PH + VGAP)
            ax = fig.add_axes([fx(LEFT), fy(y0 + PH), pw / W, PH / Hfig])
            ax.fill_between([0, 1], l2, h2, color=BAND[95], lw=0, zorder=0)
            ax.fill_between([0, 1], l1, h1, color=BAND[68], lw=0, zorder=0)
            ax.axhline(1.0, color=pc.MUTED, lw=0.4, ls=aa.DASH, zorder=1)
            ax.plot(*bias, ls=BIAS_LS, lw=GUIDE_LW, color=GUIDE_COLOR, zorder=6)
            ax.plot(*under, ls=UNDER_LS, lw=GUIDE_LW, color=GUIDE_COLOR, zorder=6)
            for i, k in enumerate(keys):
                h = H[k][j]
                if True:
                    ax.stairs(h, edges, lw=al.line_width(k, LW), color=al.line_colour(k),
                              ls=al.linestyle(k), baseline=None, joinstyle="miter",
                              capstyle="butt", zorder=al.line_z(k, 5 + 0.1 * i))
            ax.set_xlim(0, 1); ax.set_ylim(ylo, yhi)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            yt = [v for v in Y_TICKS if ylo + 0.03 < v < yhi - 0.03]
            ax.set_yticks(yt)
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params(labelsize=FS["tick"], labelbottom=(j == NPAR - 1), pad=aa.AX["tickpad"])
            ax.text(0.015, 0.94, rf"$\mathbf{{({'abcd'[j]})}}$ {PLAB[j]}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=FS["panel"],
                    zorder=8, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))
            if j == NPAR - 1:
                ax.set_xlabel("rank CDF value", fontsize=FS["label"], labelpad=aa.AX["labelpad"])
        y_mid = TOP + (NPAR * PH + (NPAR - 1) * VGAP) / 2
        fig.canvas.draw()
        x_tick = min(t.get_window_extent().x0 for a_ in fig.axes for t in a_.get_yticklabels()
                     if t.get_visible() and t.get_text())
        x_lab = x_tick / fig.dpi * 72 - aa.AX["labelpad"]
        fig.text(x_lab / (W / 25.4 * 72), fy(y_mid), "normalised rank density", rotation=90,
                 ha="right", va="center", fontsize=FS["label"], color=pc.INK)
        y_leg = TOP + NPAR * PH + (NPAR - 1) * VGAP + XLAB + LEG_GAP
        al.legend_below(fig, W, Hfig, LEFT, W - RIGHT, y_leg, cols, lw=LW,
                        col_w=LEG_COL_W)
        aa.save(fig, NAME, formats=("pdf", "svg", "png"), outdir=IMAGES)
        plt.close(fig)


def main():
    aa.use()
    H, n_used = load()
    plot(H, n_used)


if __name__ == "__main__":
    main()
