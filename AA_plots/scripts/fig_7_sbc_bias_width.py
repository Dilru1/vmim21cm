#!/usr/bin/env python3
"""Paper Fig. 7: SBC bias b and width ratio s of every configuration (Table tab:results set), one ..."""
import csv
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
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, save
import paper_colors as pc  # noqa: E402        the paper colour scheme
import arm_look as al  # noqa: E402            arm colours / shapes / runs (shared)
ROOT = HERE.parents[1]
IMAGES = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images"))
NAME = "fig_7_sbc_bias_width"

SOURCES = {k: v for k, v in al.SOURCES.items() if k != "gmm"}
GMM_CACHE = ROOT / "paper_out" / "sbc_scores" / "sbc_pulls_runs.csv"
BS_COLS = ["Fx", "tau", "rhs", "Mmin"]
CACHE_PARAMS = ["fx", "tau", "rhs", "mmin"]
PARAMS = ["log10_fX", "tau", "rHS", "log10_Mmin"]
PLAB = [r"$\log_{10}f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$", r"$\log_{10}M_{\rm min}$"]

REF_STYLE = {"ref": ("black", "*"), "gmm": ("black", "X")}
MS = {"o": 3.6, "s": 3.4, "^": 4.2, "*": 7.0, "X": 5.2}
EDGE, EDGE_W = pc.INK, 0.45
OPEN_MEW = 0.9
REF_MEC, REF_MEW, REF_ELW = "white", 0.5, 0.8
LEVELS = (1.26, 1.47, 2.5)
REGION = [pc.REGIONS_SEQ[k] for k in ("calibrated", "marginal", "miscal", "strong")]
CONTOUR_COLOR, GUIDE_COLOR = "0.35", pc.GUIDE
REG_LAB = [r"$\hat\chi<1.26$", r"$1.26<\hat\chi<1.47$", r"$1.47<\hat\chi<2.5$", r"$\hat\chi>2.5$"]
XLIM, YLIM = (-0.31, 0.26), (0.845, 1.30)
FS = dict(tick=aa.AX["tick"], label=aa.AX["label"], panel=aa.AX["panel"],
          leg=aa.FS["note"] + 0.2, head=aa.FS["note"] + 0.2)
B_BINS, N_TEST = 30, 908
ORDER = ["V4_25", "V8_25", "M4_25", "V4_50", "V8_50", "M4_50",
         "A_k16", "A_k8", "A_k4", "A_mse4", "V4_100", "V8_100", "M4_100", "gmm", "ref"]


def style(key):
    """(colour, marker, filled) of a key: sty/arm_look.py (shared with fig 6 / fig B1)."""
    return al.colour(key), al.marker(key), al.filled(key)


def load_runs():
    """{key: (runs [(source, group, run, n_sims)], array (runs, params, 2))}."""
    out = {}
    for key, (d, grp, *drop) in SOURCES.items():
        path = Path(d) / "bs_table.csv"
        rows = [r for r in csv.DictReader(open(path))
                if r["group"] == grp and not any(x in f" {r['run']} " for x in drop)]
        if not rows:
            raise SystemExit(f"no runs for {key}: {path} group {grp}")
        bs = np.array([[(float(r[f"b_{c}"]), float(r[f"s_{c}"])) for c in BS_COLS]
                       for r in rows])
        out[key] = ([(str(path.relative_to(ROOT)), grp, r["run"], r["n_sims"]) for r in rows], bs)
        print(f"  {key:5s} {grp} ({len(rows)} runs, n_sims {sorted({r['n_sims'] for r in rows})})")
    by = {}
    for r in csv.DictReader(open(GMM_CACHE)):
        if r["config"] == "gmm":
            by.setdefault(r["run"], {})[r["param"]] = (float(r["b"]), float(r["s"]))
    out["gmm"] = ([(str(GMM_CACHE.relative_to(ROOT)), "", n, "") for n in by],
                  np.array([[by[n][p] for p in CACHE_PARAMS] for n in by]))
    print(f"  gmm   csv cache ({len(by)} runs)")
    return out


def write_data(runs, stem):
    """data_runs.csv (every run) and data_points.csv (mean +- std ddof 1 = plotted)."""
    with open(f"{stem}_data_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "source", "group", "run", "n_sims", "param", "b", "s"])
        for k, (info, bs) in runs.items():
            for (src, grp, run, n), e in zip(info, bs):
                for j, par in enumerate(PARAMS):
                    w.writerow([k, src, grp, run, n, par, f"{e[j, 0]:.5f}", f"{e[j, 1]:.5f}"])
    with open(f"{stem}_data_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "n_runs", "param", "b_mean", "b_std", "s_mean", "s_std"])
        for k, (_, e) in runs.items():
            for j, par in enumerate(PARAMS):
                m = e[:, j].mean(0)
                sd = e[:, j].std(0, ddof=1) if len(e) > 1 else np.zeros(2)
                w.writerow([k, len(e), par, f"{m[0]:.5f}", f"{sd[0]:.5f}",
                            f"{m[1]:.5f}", f"{sd[1]:.5f}"])
    print(f"wrote {stem}_data_{{runs,points}}.csv")


def load_points(path):
    D = {}
    for r in csv.DictReader(open(path)):
        D.setdefault(r["key"], {})[r["param"]] = tuple(
            float(r[c]) for c in ("b_mean", "b_std", "s_mean", "s_std"))
    return D


def chi_grid(n=301):
    """E[chi_hat](b, s): rank histogram of a posterior shifted by b and scaled by s."""
    b = np.linspace(-0.45, 0.45, n)
    s = np.exp(np.linspace(np.log(0.8), np.log(1.45), n))
    bb, ss = np.meshgrid(b, s)
    e = norm.ppf(np.linspace(0, 1, B_BINS + 1))
    p = np.diff(norm.cdf(ss[..., None] * e + bb[..., None]), axis=-1)
    chi = (B_BINS * (p * (1 - p)).sum(-1)
           + N_TEST * B_BINS * ((p - 1 / B_BINS) ** 2).sum(-1)) / (B_BINS - 1)
    return bb, ss, chi


def marker(x, y, key, transform=None, **kw):
    c, mk, filled = style(key)
    if key in REF_STYLE:
        kw2 = dict(ls="none", marker=mk, ms=MS[mk], mfc=c, mec=REF_MEC, mew=REF_MEW)
    elif filled:
        kw2 = dict(ls="none", marker=mk, ms=MS[mk], mfc=c, mec=EDGE, mew=EDGE_W)
    else:
        kw2 = dict(ls="none", marker=mk, ms=MS[mk], mfc="white", mec=c, mew=OPEN_MEW)
    if transform is not None:
        kw2["transform"] = transform
    kw2.update(kw)
    return Line2D([x], [y], **kw2)


TAB_LAB_FRAC = 0.42
TAB_TINT = 0.18
TAB_HEAD_FILL = "#E9E9E9"
TAB_FS = aa.AX["table"]


def table_legend(fig, W, H, x0, y0, x1, y1, D):
    """Legend table in the box (x0, y0)-(x1, y1) pt (y from the top): rows = groups PDF+PS / Path A / Path ..."""
    from matplotlib.patches import Rectangle as Rect
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / H)
    rows = [("khdr", None, ""),
            ("head", None, "PDF+PS"),
            ("row", {30: "ref"}, "NSF"), ("row", {30: "gmm"}, "GMM"),
            ("head", None, "Path A (MLP)"),
            ("row", {4: "A_k4", 8: "A_k8", 16: "A_k16"}, "VMIM, 100 h"),
            ("row", {4: "A_mse4"}, "MSE, 100 h"),
            ("head", None, "Path B (CNN)"),
            *[("row", {4: f"V4_{h}", 8: f"V8_{h}"}, f"VMIM, {h} h") for h in (100, 50, 25)],
            *[("row", {4: f"M4_{h}"}, f"MSE, {h} h") for h in (100, 50, 25)]]
    rows = [r for r in rows if r[0] != "row" or set(r[1].values()) & set(D)]
    w, h = x1 - x0, y1 - y0
    rh = h / len(rows)
    lab_w = TAB_LAB_FRAC * w
    kw = (w - lab_w) / 4
    cx = {k: x0 + lab_w + (i + 0.5) * kw for i, k in enumerate((4, 8, 16, 30))}

    def rect(x, y, ww, hh, fc, ec="none", lw=0.0, z=1.0):
        fig.add_artist(Rect((fx(x), fy(y + hh)), ww / W, hh / H, fc=fc, ec=ec, lw=lw, zorder=z,
                            transform=fig.transFigure))

    rect(x0, y0, w, h, "white", z=0.9)
    for i, (kind, ks, lab) in enumerate(rows):
        y = y0 + i * rh
        yc = y + 0.5 * rh
        if kind in ("khdr", "head"):
            rect(x0, y, w, rh, TAB_HEAD_FILL)
            if kind == "khdr":
                fig.text(fx(x0 + 2.5), fy(yc), r"$k$", ha="left", va="center_baseline",
                         fontsize=TAB_FS, zorder=3)
                for k, c in cx.items():
                    fig.text(fx(c), fy(yc), f"{k}", ha="center", va="center_baseline",
                             fontsize=TAB_FS, zorder=3)
            else:
                fig.text(fx(x0 + 2.5), fy(yc), lab, ha="left", va="center_baseline",
                         fontsize=TAB_FS, style="italic", weight="bold", zorder=3)
            continue
        fig.text(fx(x0 + 5.0), fy(yc), lab, ha="left", va="center_baseline", fontsize=TAB_FS,
                 zorder=3)
        for k, key in ks.items():
            if key not in D:
                continue
            c = cx[k]
            rect(c - 0.5 * kw, y, kw, rh, pc._on_white(style(key)[0], TAB_TINT), z=1.5)
            col = style(key)[0]
            fig.add_artist(Line2D([fx(c - 0.36 * kw), fx(c + 0.36 * kw)], [fy(yc)] * 2,
                                  color=col, lw=REF_ELW if key in REF_STYLE else 0.6,
                                  transform=fig.transFigure, zorder=3))
            fig.add_artist(marker(fx(c), fy(yc), key, transform=fig.transFigure, zorder=4,
                                  ms=0.8 * MS[style(key)[1]]))
    rect(x0, y0, w, h, "none", ec=mpl.colors.to_rgba(pc.INK, 0.3), lw=aa.LW["axes"], z=5)


def plot(D, stem):
    order = [k for k in ORDER if k in D]
    bb, ss, chi = chi_grid()

    W = 521.28
    left, gap, leg_w, right = 27.0, 2.5, 96.0, 1.0
    pw = (W - left - leg_w - right - 3 * gap - 4.0) / 4
    ph = pw
    top_leg, xlab = 10.0, 19.0
    H = top_leg + ph + xlab
    fx = lambda x: x / W                             # noqa: E731  pt -> figure fraction
    fy = lambda y: 1 - y / H                         # noqa: E731  (y from the top)
    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W / 72, H / 72))
        for j, (par, lab) in enumerate(zip(PARAMS, PLAB)):
            ax = fig.add_axes([fx(left + j * (pw + gap)), fy(top_leg + ph), pw / W, ph / H])
            ax.contourf(bb, ss, chi, levels=[0, *LEVELS, np.inf], colors=REGION, zorder=0)
            ax.contour(bb, ss, chi, levels=LEVELS, colors=CONTOUR_COLOR, linewidths=0.45,
                       zorder=1)
            ax.axvline(0, color=GUIDE_COLOR, lw=0.35, ls=aa.DASH, zorder=0.8)
            ax.axhline(1, color=GUIDE_COLOR, lw=0.35, ls=aa.DASH, zorder=0.8)
            for i, key in enumerate(order):
                b, bs, s, sd = D[key][par]
                c = style(key)[0]
                isref = key in REF_STYLE
                z = (6 if isref else 3) + 0.05 * i
                ax.errorbar(b, s, xerr=bs, yerr=sd, fmt="none", ecolor=c,
                            elinewidth=REF_ELW if isref else 0.6,
                            capsize=1.3 if isref else 1.1,
                            capthick=REF_ELW if isref else 0.6, zorder=z)
                ax.add_line(marker(b, s, key, zorder=z + 1))
            ax.set_yscale("log")
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
            ax.yaxis.set_major_locator(FixedLocator([0.9, 1.0, 1.1, 1.2]))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.85, YLIM[1], 0.025)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([-0.2, 0.0, 0.2])
            ax.xaxis.set_minor_locator(FixedLocator(np.arange(-0.25, 0.26, 0.05)))
            ax.tick_params(labelsize=FS["tick"], labelleft=(j == 0), pad=aa.AX["tickpad"])
            ax.set_xlabel(r"bias $b$ ($\sigma$)", fontsize=FS["label"], labelpad=aa.AX["labelpad"])
            if j == 0:
                ax.set_ylabel(r"width ratio $s$", fontsize=FS["label"], labelpad=aa.AX["labelpad"])
            ax.text(0.04, 0.96, rf"$\mathbf{{({'abcd'[j]})}}$ {lab}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=FS["panel"], zorder=8)

        rend = fig.canvas.get_renderer()

        def tw(txt, fs):
            t = fig.text(0, 0, txt, fontsize=fs)
            w = t.get_window_extent(rend).width * 72 / fig.dpi
            t.remove()
            return w

        hw, pad = 9.0, 2.0
        x_a, x_b = left, left + 4 * pw + 3 * gap
        ws = [hw + pad + tw(t, FS["leg"]) for t in REG_LAB]
        g = (x_b - x_a - sum(ws)) / (len(ws) - 1)
        x, y = x_a, top_leg / 2 + 0.3
        for v, w in enumerate(ws):
            fig.add_artist(Rectangle((fx(x), fy(y + 2.6)), hw / W, 5.2 / H, fc=REGION[v],
                                     ec=CONTOUR_COLOR, lw=0.45, transform=fig.transFigure))
            fig.text(fx(x + hw + pad), fy(y), REG_LAB[v], ha="left", va="center_baseline",
                     fontsize=FS["leg"])
            x += w + g

        table_legend(fig, W, H, left + 4 * pw + 3 * gap + 4.0, top_leg, W - right, top_leg + ph, D)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)
    print(f"wrote {stem}.{{pdf,svg,png}}  ({W / 72 * 25.4:.0f} x {H / 72 * 25.4:.1f} mm)")


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    stem = str(IMAGES / NAME)
    print("(b, s) per run:")
    write_data(load_runs(), stem)
    plot(load_points(f"{stem}_data_points.csv"), stem)


if __name__ == "__main__":
    main()
