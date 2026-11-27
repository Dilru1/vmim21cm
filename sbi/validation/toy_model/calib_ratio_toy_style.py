"""Figure of calib_ratio_toy.py: toy test sets in the (z_bias, z_width) plane."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CHI_CAL, CHI_MIS = 1.26, 1.47
BAND = "#4C72B0"
import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _palette import TOY_LOOK as LOOK  # noqa: E402
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif"})


def draw_plane(ax, contours, lim=8.0, fs=7, contour_fs=6.5,
               contour_pos=((-1.9, 1.8), (2.5, 2.5)), calib_pos=(0.5, 1.5)):
    """chi_hat contours of the Gaussian model (labelled on the boundary), axes through zero and region ..."""
    zz, chi = contours
    ax.tricontourf(zz[:, 0], zz[:, 1], chi, levels=[0, CHI_CAL], colors=[BAND], alpha=0.15)
    cs = [ax.tricontour(zz[:, 0], zz[:, 1], chi, levels=[CHI_CAL], colors=BAND, linewidths=0.9),
          ax.tricontour(zz[:, 0], zz[:, 1], chi, levels=[CHI_MIS], colors=BAND, linewidths=0.8,
                        linestyles="--")]
    for c, lev, pos in zip(cs, (CHI_CAL, CHI_MIS), contour_pos):
        if contour_fs:
            ax.clabel(c, fmt={lev: rf"$\hat\chi={lev}$"}, manual=[pos], inline=True,
                      fontsize=contour_fs)
    ax.axvline(0, color="0.8", lw=0.4)
    ax.axhline(0, color="0.8", lw=0.4)
    for t, x, y, ha in [("under-confident", 0, lim - 0.6, "center"), ("over-confident", 0, 0.4 - lim, "center"),
                        ("high bias", lim - 0.4, 0.4, "right"), ("low bias", 0.4 - lim, 0.4, "left"),
                        ("calibrated", *(calib_pos or (None, None)), "center")]:
        if x is None:
            continue
        ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color="k")


def plot_scores(scores, expected, contours, out):
    """scores[key]: (seeds, 2) per test set; expected[key]: (2,) expected scores; contours: (expected ..."""
    fig, ax = plt.subplots(figsize=(3.54, 3.54), layout="constrained")

    draw_plane(ax, contours)

    for key, (label, c, mk) in LOOK.items():
        z, e = scores[key], expected[key]
        ax.scatter(z[:, 0], z[:, 1], s=4, color=c, alpha=0.35, lw=0, zorder=3)
        ax.plot(*e, mk, ms=6.5, mfc="none", mec=c, mew=1.0, zorder=4)
        ax.plot(*z.mean(0), mk, ms=4.5, color=c, mec="white", mew=0.5, zorder=5)
    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="k"),
               Line2D([], [], ls="", marker="o", ms=4.5, color="k", mec="white", mew=0.5)]
    labels = ["true value", "expected value"]

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r"bias score")
    ax.set_ylabel(r"width score")
    ax.legend(handles, labels, loc="lower left", frameon=False, handletextpad=0.3)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=250)
    plt.close(fig)
