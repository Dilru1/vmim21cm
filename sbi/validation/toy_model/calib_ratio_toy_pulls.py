#!/usr/bin/env python3
"""Gaussian toy model: bias and width measured from the ranks of all N test simulations."""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

import calib_ratio_toy as toy

N, SEEDS = toy.N, 9
LEVELS = (1.26, 1.47, 2.5)
BAND, LW = "k", 0.5
BLIM, SLIM = (-0.3, 0.3), (0.75, 1.35)
B_TICKS, S_TICKS = (0.2, 0.05), (0.2, 0.05)
XLABEL, YLABEL = r"bias $b$ ($\sigma$)", r"width ratio $s$"
REGIONS = [
    ("calibrated", r"$\hat\chi<1.26$", "#D3E8F5"),
    ("marginal", r"$1.26<\hat\chi<1.47$", "#E4D8F0"),
    ("miscalibrated", r"$1.47<\hat\chi<2.5$", "#F1E5C8"),
    ("strongly miscalibrated", r"$\hat\chi>2.5$", "white"),
]
TRUE_BS = {
    "cal": (0.0, 1.0), "bias": (0.15, 1.0), "over": (0.0, 0.88), "under": (0.0, 1.15),
    "both": (0.12, 0.92), "theta": (0.0, 1 / np.sqrt(1 + 0.3 ** 2)), "skew": (0.0, 1.0)}
import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _palette import TOY_LOOK as LOOK  # noqa: E402
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "legend.fontsize": 6.3})


def gauss_ranks(truth, post):
    """u_a = Phi^-1(r_a), r_a = fraction of posterior samples below the truth (as eval.py), with (count + ..."""
    m = post.shape[1]
    return norm.ppf(((post < truth[:, None]).sum(1) + 0.5) / (m + 1))


def bs_from_u(u):
    sd = u.std(ddof=1)
    return -u.mean() / sd, 1.0 / sd


def bs_meanstd(truth, post):
    """the same from the posterior mean and std: (theta_true - mu_a) / sigma_a."""
    return bs_from_u((truth - post.mean(1)) / post.std(1))


def chi_grid(bmax=0.3, smin=0.7, smax=1.3, n=81):
    b, s = np.linspace(-bmax, bmax, n), np.linspace(smin, smax, n)
    chi = np.array([[toy.expected_chi(toy.bin_probs(toy.gauss(bj, si)[1])) for bj in b]
                    for si in s])
    return b, s, chi


def toy_bs(key, seeds=SEEDS):
    """(b_hat, s_hat) of `seeds` test sets of one case, (seeds, 2)."""
    i = list(toy.CASES).index(key)
    draw = toy.CASES[key][0]
    return np.array([bs_from_u(gauss_ranks(*draw(np.random.default_rng(10000 * (i + 1) + k))))
                     for k in range(seeds)])


def seed_colour(c, f=0.3):
    """marker colour darkened by f towards black: the + of single test sets / seeds."""
    return tuple((1 - f) * np.array(to_rgb(c)))


def region_handles(names=True):
    """legend patches of the regions; names=False: chi_hat ranges only."""
    return [Patch(fc=c, ec="k", lw=0.4, label=f"{n}, {r}" if names else r) for n, r, c in REGIONS]


def draw_bs_plane(ax, grid, fs=6.5, contour_fs=5, blim=BLIM, slim=SLIM, side="right",
                  region_text=True, guide_kw=None):
    """E[chi_hat](b, s) of the Gaussian model as shaded regions (REGIONS) with thin contours, axes through ..."""
    b, s, chi = grid
    ax.contourf(b, s, chi, levels=[0, *LEVELS, np.inf], colors=[c for *_, c in REGIONS],
                zorder=0)
    cs = ax.contour(b, s, chi, levels=LEVELS, colors=BAND, linewidths=LW, linestyles="-",
                    zorder=1)
    if contour_fs:
        sg = 1 if side == "right" else -1
        ax._bs_labels = [(lev, max(segs, key=len), (sg * x, y), contour_fs)
                         for lev, segs, (x, y) in zip(LEVELS, cs.allsegs,
                                                      ((0.08, 1.05), (0.11, 0.93), (0.21, 1.05)))]
    for sp in ax.spines.values():
        sp.set_linewidth(LW)
    ax.tick_params(width=LW)
    if guide_kw is not False:
        gk = dict(color="0.8", lw=0.4) if guide_kw is None else guide_kw
        ax.axvline(0, **gk)
        ax.axhline(1, **gk)
    ax.set_xlim(*blim)
    ax.set_ylim(*slim)
    bs_axes(ax)
    if not region_text:
        return
    xr = blim[1] - 0.02 * (blim[1] - blim[0])
    ax.text(xr, slim[1] - 0.01, "too wide", ha="right", va="top", fontsize=fs)
    ax.text(xr, slim[0] + 0.01, "too narrow", ha="right", va="bottom", fontsize=fs)


def bs_axes(ax):
    """Square panel, same major / minor ticks in every (b, s) figure, ticks inside on all four sides."""
    from matplotlib.ticker import MultipleLocator
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(B_TICKS[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(B_TICKS[1]))
    ax.yaxis.set_major_locator(MultipleLocator(S_TICKS[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(S_TICKS[1]))
    ax.tick_params(which="both", direction="in", top=True, right=True, width=LW)
    ax.tick_params(which="major", length=3.0)
    ax.tick_params(which="minor", length=1.6)


def label_contours(fig, gap=1.0):
    """Write the chi_hat labels of draw_bs_plane just outside their (unbroken) contours, along the line ..."""
    fig.canvas.draw()
    for ax in fig.axes:
        for lev, seg, target, fs in getattr(ax, "_bs_labels", []):
            d = ax.transData.transform(seg)
            i = np.argmin(np.hypot(*(d - ax.transData.transform(target)).T))
            t = d[(i + 1) % len(d)] - d[i - 1]
            n = np.array([-t[1], t[0]]) / np.hypot(*t)
            if n @ (d[i] - ax.transData.transform((0.0, 1.0))) < 0:
                n = -n
            ang = np.degrees(np.arctan2(t[1], t[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
            ax.annotate(rf"$\hat\chi={lev}$", seg[i], xytext=n * (gap + 0.6 * fs),
                        textcoords="offset points", rotation=ang, rotation_mode="anchor",
                        ha="center", va="center", fontsize=fs, zorder=2, in_layout=False)


FIG_RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
}
FIG_REGION = ["#CFE4F4", "#E3D7EF", "#F7E4C0", "white"]
FIG_REG_LAB = [r"$\hat\chi<1.26$", r"$1.26<\hat\chi<1.47$", r"$1.47<\hat\chi<2.5$",
               r"$\hat\chi>2.5$"]
FIG_COL = {"cal": (0.0, 0.0, 0.0), "bias": (0.62, 0.0, 0.086), "over": (0.031, 0.188, 0.42),
           "under": (0.0, 0.267, 0.106), "both": (0.941, 0.353, 0.055),
           "theta": (0.478, 0.196, 0.596), "skew": (0.0, 0.71, 0.847)}
FIG_MS = {"o": 4.6, "D": 4.4, "s": 4.4, "^": 5.2, "v": 5.2, "p": 5.2}
FIG_BLIM, FIG_SLIM = (-0.30, 0.30), (0.68, 1.42)


def main(fig_out=None):
    est = {}
    for key in toy.CASES:
        est[key] = toy_bs(key)
        m, sd = est[key].mean(0), est[key].std(0, ddof=1)
        print(f"{key:>6s}  b = {m[0]:+.3f} +- {sd[0]:.3f} (true {TRUE_BS[key][0]:+.3f})   "
              f"s = {m[1]:.3f} +- {sd[1]:.3f} (true {TRUE_BS[key][1]:.3f})")
    from matplotlib.ticker import MultipleLocator
    b_g, s_g, chi = chi_grid(smin=FIG_SLIM[0], smax=FIG_SLIM[1])
    with plt.rc_context(FIG_RC):
        fig, ax = plt.subplots(figsize=(3.54, 3.54), layout="constrained")
        ax.contourf(b_g, s_g, chi, levels=[0, *LEVELS, np.inf], colors=FIG_REGION, zorder=0)
        ax.contour(b_g, s_g, chi, levels=LEVELS, colors="0.25", linewidths=0.5, zorder=1)
        ax.axvline(0, color="0.55", lw=0.4, ls=(0, (3, 2)), zorder=0.8)
        ax.axhline(1, color="0.55", lw=0.4, ls=(0, (3, 2)), zorder=0.8)
        for i, (key, (lab, _, mk)) in enumerate(LOOK.items()):
            e, c = est[key], FIG_COL[key]
            m, sd = e.mean(0), e.std(0, ddof=1)
            ax.plot(*TRUE_BS[key], mk, ms=FIG_MS.get(mk, 4.6) + 3.2, mfc="none", mec=c,
                    mew=0.9, zorder=4)
            ax.errorbar(m[0], m[1], xerr=sd[0], yerr=sd[1], fmt="none", ecolor=c,
                        elinewidth=0.75, capsize=1.6, capthick=0.75, zorder=5 + 0.1 * i)
            ax.plot(*m, mk, ms=FIG_MS.get(mk, 4.6), mfc=c, mec="k", mew=0.5,
                    zorder=6 + 0.1 * i)
        ax.set_xlim(*FIG_BLIM)
        ax.set_ylim(*FIG_SLIM)
        ax.set_box_aspect(1)
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.set_yticks([0.8, 1.0, 1.2, 1.4])
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlabel(XLABEL, fontsize=9.5, labelpad=1.5)
        ax.set_ylabel(YLABEL, fontsize=9.5, labelpad=2.0)
        ax.tick_params(labelsize=8.0, pad=2.5)
        h = [Line2D([], [], ls="-", lw=0.75, color=FIG_COL[k], marker=mk,
                    ms=FIG_MS.get(mk, 4.6), mfc=FIG_COL[k], mec="k", mew=0.5)
             for k, (_, _, mk) in LOOK.items()]
        labs = [l for l, _, _ in LOOK.values()]
        h += [Line2D([], [], ls="", marker="o", ms=7.5, mfc="none", mec="k", mew=0.9),
              Line2D([], [], ls="-", lw=0.75, color="0.4", marker="o", ms=4.6, mfc="0.6",
                     mec="k", mew=0.5)]
        labs += ["true value", f"mean $\\pm$ std, {SEEDS} test sets"]
        leg = ax.legend(h, labs, loc="upper left", ncol=2, fontsize=5.0, frameon=True,
                        framealpha=0.9, edgecolor="none", fancybox=False, handlelength=1.6,
                        handletextpad=0.4, columnspacing=0.8, labelspacing=0.25,
                        borderpad=0.3, borderaxespad=0.3)
        ax.add_artist(leg)
        rh = [Patch(fc=c, ec="0.25", lw=0.5) for c in FIG_REGION]
        ax.legend(rh, FIG_REG_LAB, loc="lower center", ncol=4, fontsize=5.0, frameon=False,
                  handlelength=1.2, handleheight=0.8, handletextpad=0.3, columnspacing=0.6,
                  borderaxespad=0.3)
        fig_out = fig_out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                          "..", "..", "paper_plot_final", "images",
                                          "validation")
        os.makedirs(fig_out, exist_ok=True)
        out = os.path.join(fig_out, "calib_ratio_toy_pulls.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".svg"))
        fig.savefig(out.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
