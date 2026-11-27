#!/usr/bin/env python3
"""aa_style.py -- the A&A figure style of the SBC calibration figures (sbc_full)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FuncFormatter

MM = 1 / 25.4
WIDTH_MM = {"column": 90.0, "full": 184.0, "side": 120.0}

FONT_SERIF = ["STIXGeneral", "STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"]
MATHTEXT = "stix"

FS = {
    "label": 8.5,
    "tick": 7.0,
    "panel": 7.5,
    "legend": 6.8,
    "note": 6.0,
}

LW = {
    "axes": 0.6,
    "contour": 0.5,
    "guide": 0.5,
    "edge": 0.5,
    "cross": 0.6,
    "legend_box": 0.4,
}
DASH = (0, (3, 2))

TICK = {
    "direction": "in", "all_sides": True,
    "major": 2.8, "minor": 1.5,
    "pad": 2.0,
}

INK = "#1A1A1A"
MUTED = "0.42"
GUIDE = "black"

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _palette import ARMS as _ARMS  # noqa: E402
CONFIG = {k: (mpl.colors.to_rgb(c), m) for k, (_, c, m) in _ARMS.items()}

REGION = {
    "calibrated": (0.827, 0.910, 0.961),
    "marginal":   (0.894, 0.847, 0.941),
    "miscal":     (0.945, 0.898, 0.784),
    "strong":     (1.000, 1.000, 1.000),
}

TOY_COLOR = "0.45"
CROSS_DARKEN = 0.3

MS = {"o": 4.6, "s": 4.4, "D": 4.0, "^": 4.8, "v": 4.8}
MS_CROSS = 3.2
MS_TOY = 1.6
TRUE_SCALE = 1.45


def darken(c, a=0.3):
    r, g, b = mpl.colors.to_rgb(c)
    return (r * (1 - a), g * (1 - a), b * (1 - a))


def marker(color, shape="o", **kw):
    """Filled configuration marker with a thin black edge (plot kwargs)."""
    d = dict(ls="none", marker=shape, ms=MS[shape], mfc=color, mec="black",
             mew=LW["edge"], zorder=6)
    d.update(kw)
    return d


def cross(color, **kw):
    """'+' for one run / seed, in a darker shade of its configuration."""
    d = dict(ls="none", marker="+", ms=MS_CROSS, mew=LW["cross"],
             color=darken(color, CROSS_DARKEN), zorder=3)
    d.update(kw)
    return d


def open_marker(color, shape="o", **kw):
    """Large open marker for a true value."""
    d = dict(ls="none", marker=shape, ms=MS[shape] * TRUE_SCALE, mfc="none", mec=color,
             mew=1.1, zorder=4)
    d.update(kw)
    return d


def toy(**kw):
    d = dict(ls="none", marker="o", ms=MS_TOY, mfc=TOY_COLOR, mec="none", alpha=0.8,
             zorder=2)
    d.update(kw)
    return d


MARKER = {k: marker(c, m) for k, (c, m) in CONFIG.items()}

def rc():
    return {
        "font.family": "serif", "font.serif": FONT_SERIF,
        "mathtext.fontset": MATHTEXT, "text.usetex": False,
        "font.size": FS["tick"],
        "axes.labelsize": FS["label"], "axes.titlesize": FS["panel"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "axes.unicode_minus": True,
        "axes.formatter.use_mathtext": True,
        "axes.linewidth": LW["axes"], "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.labelpad": 2.0,
        "xtick.direction": TICK["direction"], "ytick.direction": TICK["direction"],
        "xtick.top": TICK["all_sides"], "ytick.right": TICK["all_sides"],
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": TICK["major"], "ytick.major.size": TICK["major"],
        "xtick.minor.size": TICK["minor"], "ytick.minor.size": TICK["minor"],
        "xtick.major.width": LW["axes"], "ytick.major.width": LW["axes"],
        "xtick.minor.width": 0.75 * LW["axes"], "ytick.minor.width": 0.75 * LW["axes"],
        "xtick.major.pad": TICK["pad"], "ytick.major.pad": TICK["pad"],
        "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 1.0, "lines.markersize": MS["o"],
        "lines.markeredgewidth": LW["edge"],
        "patch.linewidth": LW["contour"],
        "axes.prop_cycle": cycler(color=[c for c, _ in CONFIG.values()][1:]),
        "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4, "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0,
        "savefig.dpi": 600, "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.interpolation": "nearest",
    }


def use():
    mpl.rcParams.update(rc())


def panels(n=4, width="full", left=9.6, gap=1.7, right=0.3, top=0.6, bottom=7.8,
           legend_mm=0.0, legend_gap=2.6, aspect=1.0):
    """n square panels in a row, an optional legend column on the right."""
    W = WIDTH_MM[width] if isinstance(width, str) else float(width)
    pw = (W - left - (n - 1) * gap - (legend_gap + legend_mm if legend_mm else 0)
          - right) / n
    ph = pw * aspect
    H = top + ph + bottom
    fig = plt.figure(figsize=(W * MM, H * MM))
    axes = [fig.add_axes([(left + i * (pw + gap)) / W, bottom / H, pw / W, ph / H])
            for i in range(n)]
    fig.aa = dict(W=W, H=H, pw=pw, ph=ph, left=left, gap=gap, top=top, n=n,
                  legend_x=left + n * pw + (n - 1) * gap + legend_gap, legend_w=legend_mm)
    for i, ax in enumerate(axes):
        if i:
            ax.tick_params(labelleft=False)
    return fig, axes


def shared_xlabel(fig, text, y_mm=0.9):
    g = fig.aa
    xc = g["left"] + (g["n"] * g["pw"] + (g["n"] - 1) * g["gap"]) / 2
    fig.text(xc / g["W"], y_mm / g["H"], text, ha="center", va="bottom",
             fontsize=FS["label"], color=INK)


def zero_cross(ax, x0=0.0, y0=None):
    """Black dashed reference lines (b = 0, s = 1)."""
    kw = dict(color=GUIDE, lw=LW["guide"], ls=DASH, zorder=1.5)
    if x0 is not None:
        ax.axvline(x0, **kw)
    if y0 is not None:
        ax.axhline(y0, **kw)


def minor_ticks(ax, x=None, y=None):
    if x:
        ax.xaxis.set_minor_locator(MultipleLocator(x))
    if y:
        ax.yaxis.set_minor_locator(MultipleLocator(y))


def clean_zero(ax, axis="x", fmt="{:.1f}"):
    """Tick labels '-0.2  0  0.2' (bare 0, maths minus)."""
    f = FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else rf"${fmt.format(v)}$")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def panel_label(ax, letter, text="", loc="upper left"):
    """(a) + parameter, top left, on a light white box so data never cut it."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    s = rf"$\mathbf{{({letter})}}$ {text}".strip()
    return ax.text(x, 0.965, s, transform=ax.transAxes, ha=ha, va="top",
                   fontsize=FS["panel"], color=INK, zorder=8,
                   bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def note(ax, text, loc="upper right", italic=False):
    """Small grey note in a corner ("too wide", "too narrow")."""
    x, ha = (0.96, "right") if "right" in loc else (0.04, "left")
    y, va = (0.965, "top") if "upper" in loc else (0.035, "bottom")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=FS["note"], color=MUTED, zorder=8,
                   style="italic" if italic else "normal",
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))


class GridLegend:
    """Frameless legend drawn on a mm grid at the right of the panels, so every marker and every text ..."""

    def __init__(self, fig, row=2.72, y0=1.2, text_dx=3.6, marker_dx=1.4):
        g = fig.aa
        self.fig, self.W, self.H = fig, g["W"], g["H"]
        self.x, self.w = g["legend_x"], g["legend_w"]
        self.y = g["top"] + y0
        self.dy, self.tdx, self.mdx = row, text_dx, marker_dx

    def _fx(self, x): return x / self.W
    def _fy(self, y): return 1 - y / self.H

    def _mark(self, x, kw):
        kw = {k: v for k, v in kw.items() if k not in ("zorder",)}
        self.fig.add_artist(Line2D([self._fx(x + self.mdx)], [self._fy(self.y)], **kw))

    def _text(self, x, s, **kw):
        self.fig.text(self._fx(x + self.tdx), self._fy(self.y), s, ha="left",
                      va="center_baseline", fontsize=FS["legend"], color=INK, **kw)

    def marker(self, label, **kw):
        self._mark(self.x, kw); self._text(self.x, label); self.y += self.dy

    def row(self, items):
        n = len(items)
        for k, (label, kw) in enumerate(items):
            x = self.x + k * self.w / n
            self._mark(x, kw); self._text(x, label)
        self.y += self.dy

    def header(self, text, space=0.12):
        self.y += space * self.dy
        self.fig.text(self._fx(self.x), self._fy(self.y), text, ha="left",
                      va="center_baseline", fontsize=FS["legend"], style="italic", color=INK)
        self.y += self.dy

    def patch(self, label, fc):
        self.fig.add_artist(Rectangle((self._fx(self.x + 0.3), self._fy(self.y + 0.95)),
                                      2.2 / self.W, 1.9 / self.H, fc=fc, ec="black",
                                      lw=LW["legend_box"], transform=self.fig.transFigure))
        self._text(self.x, label); self.y += self.dy

    def space(self, f=0.25):
        self.y += f * self.dy


def save(fig, name, formats=("pdf", "svg"), outdir="."):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(outdir / f"{name}.{f}")
    w, h = fig.get_size_inches() / MM
    print(f"wrote {outdir / name}.{{{','.join(formats)}}}  ({w:.0f} x {h:.1f} mm)")


def _demo():
    from matplotlib.patches import Ellipse
    use()
    rng = np.random.default_rng(1)
    params = [r"$\log_{10} f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$", r"$\log_{10} M_{\rm min}$"]
    fig, axes = panels(n=4, width="full", legend_mm=33.5)
    for p, ax in enumerate(axes):
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(0.75, 1.35)
        for (a, b), key in zip([(0.24, 0.14), (0.13, 0.085), (0.095, 0.062)],
                               ["miscal", "marginal", "calibrated"]):
            ax.add_patch(Ellipse((0, 1.0), 2 * a, 2 * b, fc=REGION[key], ec="none", zorder=0))
            ax.add_patch(Ellipse((0, 1.0), 2 * a, 2 * b, fc="none", ec="black",
                                 lw=LW["contour"], zorder=1))
        zero_cross(ax, 0, 1)
        t = rng.normal([0, 1], [0.04, 0.025], (50, 2))
        ax.plot(t[:, 0], t[:, 1], **toy())
        for key, (col, shp) in CONFIG.items():
            c = rng.normal([0, 1], [0.08, 0.08])
            s = c + rng.normal(0, [0.03, 0.03], (9 if key != "ref" else 3, 2))
            ax.plot(s[:, 0], s[:, 1], **cross(col))
            ax.plot([c[0]], [c[1]], **MARKER[key])
        panel_label(ax, "abcd"[p], params[p])
        note(ax, "too wide", "upper right"); note(ax, "too narrow", "lower right")
        ax.set_xticks([-0.2, 0.0, 0.2]); ax.set_yticks(np.arange(0.8, 1.31, 0.1))
        minor_ticks(ax, x=0.05, y=0.025); clean_zero(ax, "x")
        if p == 0:
            ax.set_ylabel(r"width ratio $s$")
    shared_xlabel(fig, r"bias $b$ [$\sigma$]")

    lg = GridLegend(fig)
    lg.marker("PDF+PS (reference)", **MARKER["ref"])
    lg.marker("one run (seed)", **cross("0.2"))
    lg.marker("toy model", **toy(ms=2.0))
    lg.header("Path A")
    lg.row([(r"$k=4$", MARKER["A_k4"]), (r"$k=8$", MARKER["A_k8"]),
            (r"$k=16$", MARKER["A_k16"])])
    lg.header("Path B")
    for v, m in (("100", "100"), ("50", "50"), ("25", "25")):
        lg.row([(f"VMIM {v} h", MARKER[f"VMIM_{v}"]), (f"MSE {m} h", MARKER[f"MSE_{m}"])])
    lg.space()
    lg.patch(r"$\hat\chi\leq1.26$", REGION["calibrated"])
    lg.patch(r"$1.26<\hat\chi\leq1.47$", REGION["marginal"])
    lg.patch(r"$1.47<\hat\chi\leq2.5$", REGION["miscal"])
    save(fig, "aa_style_demo")


if __name__ == "__main__":
    _demo()
