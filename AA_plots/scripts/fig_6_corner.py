#!/usr/bin/env python3
"""Paper Fig. 6: posterior corner plots of N_CORNER test simulations, one A&A column each."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sbi" / "validation"))
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402
import paper_colors as pc  # noqa: E402
import arm_look as al  # noqa: E402

OUT = Path(__import__("os").environ.get("AA_IMAGES", HERE.parent / "images")) / "corner"
NAME = "fig_6_corner"
KEYS = list(al.LINE_KEYS)
PICK_KEY = "ref"
N_CORNER = 10
THIN, BURNIN, DLOGP = 11, 0.0, 10.0
PARAMS = [0, 1, 2, 3]
PLAB = [r"$\log_{10}f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$", r"$\log_{10}M_{\rm min}$"]
CONTOUR_LEVEL = 0.865
LIM_Q, LIM_PAD = 0.7, 0.05
BINS, SMOOTH = 30, 1.0

LW = 0.7
TRUTH_LINE = dict(color=pc.MUTED, lw=0.45, ls=aa.DASH)
TRUTH_MK = dict(marker="s", color=pc.INK, mec="white", mew=0.5)
FS = {"tick": aa.AX["tick"], "label": aa.AX["label"]}
W = aa.WIDTH_MM["column"]
LEFT, RIGHT, TOP = 10.5, 1.5, 1.0
GAP = 1.0
XLAB = 8.0
LEG_LAB_FRAC = 0.47
LEG_TINT = 0.18
LEG_HEAD_FILL = "#E9E9E9"
LEG_FRAME_ALPHA = 0.3
LEG_SIZE = 0.82
LEG_FS = aa.AX["table"]


def load_arms():
    """{key: [(dir, n_params, {sim: file})]} for every run of every arm."""
    from eval import list_chains, chain_sim
    arms = {}
    for k in KEYS:
        arms[k] = []
        for r in al.runs(k):
            m = {chain_sim(f): f for f in list_chains(r["dir"]) if chain_sim(f) is not None}
            arms[k].append((r["dir"], r["n_params"], m))
        print(f"  {k:7s} {len(arms[k])} runs ({al.SOURCES[k][1]})")
    return arms


def pick_sims(arms, n):
    from eval import load_chain
    shared = set.intersection(*[set(m) for runs in arms.values() for _, _, m in runs])
    _, npar, m = arms[PICK_KEY][0]
    gv = {}
    for s in sorted(shared):
        ld = load_chain(m[s], npar, BURNIN, THIN, DLOGP)
        if ld is None:
            continue
        sign, ldet = np.linalg.slogdet(np.cov(ld[0][:, PARAMS], rowvar=False))
        if sign > 0 and np.isfinite(ldet):
            gv[s] = np.exp(0.5 * ldet)
    med = np.median(list(gv.values()))
    print(f"  {len(shared)} shared sims, median GV ({PICK_KEY}) {med:.3g}")
    return sorted(gv, key=lambda s: abs(gv[s] - med))[:n]


def pooled(arms, sim):
    """{key: pooled samples}, truth."""
    from eval import load_chain
    out, truth = {}, None
    for k, runs in arms.items():
        ss = []
        for _, npar, m in runs:
            ld = load_chain(m[sim], npar, BURNIN, THIN, DLOGP)
            if ld is not None:
                ss.append(ld[0])
                truth = ld[1] if truth is None else truth
        if ss:
            out[k] = np.concatenate(ss)
    return out, truth


def drawn_limits(S, q=LIM_Q, pad=LIM_PAD):
    lims = []
    for j in PARAMS:
        lo = min(np.percentile(s[:, j], q) for s in S.values())
        hi = max(np.percentile(s[:, j], 100 - q) for s in S.values())
        w = max(hi - lo, 1e-6)
        lims.append((lo - pad * w, hi + pad * w))
    return lims


def level(H, frac, total):
    flat = np.sort(H.ravel())[::-1]
    c = np.cumsum(flat) / total
    return flat[min(np.searchsorted(c, frac), len(flat) - 1)]


def contour(ax, k, x, y, xr, yr, nb, sm):
    H, xe, ye = np.histogram2d(x, y, bins=nb, range=[xr, yr])
    H = gaussian_filter(H.astype(float), sm)
    if H.sum() <= 0:
        return
    lv = level(H, CONTOUR_LEVEL, len(x))
    if not 0 < lv < H.max():
        return
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if True:
        ax.contour(xc, yc, H.T, levels=[lv], colors=[al.line_colour(k)],
                   linewidths=al.line_width(k, LW), linestyles=[al.linestyle(k)],
                   zorder=al.line_z(k, 5 + KEYS.index(k) * 0.1))


def hist(ax, k, x, rng, nb, horizontal=False):
    o = "horizontal" if horizontal else "vertical"
    if True:
        ax.hist(x, bins=nb, range=rng, density=True, histtype="step", orientation=o,
                color=al.line_colour(k), ls=al.linestyle(k), lw=al.line_width(k, LW),
                zorder=al.line_z(k, 5 + KEYS.index(k) * 0.1))


def despine(ax, horizontal=False):
    if horizontal:
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", which="both", right=False)
    else:
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", which="both", top=False)


def ticks(axis, lim, n=4, margin=0.07):
    """Round ticks, labelled with one decimal (x.x): steps of 0.1 (0.2 if crowded); 0.05 with two decimals ..."""
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    lo, hi = lim
    span = hi - lo
    for step, fmt in ((0.2, "%.1f"), (0.1, "%.1f"), (0.05, "%.2f")):
        t = np.arange(np.ceil((lo + margin * span) / step) * step, hi - margin * span + 1e-9, step)
        if 2 <= len(t) <= n:
            break
    axis.set_ticks(t)
    axis.set_major_formatter(FormatStrFormatter(fmt))
    axis.set_minor_locator(MultipleLocator(step / 2))


def table_legend(fig, Hf, x1, y0, size, keys):
    """Square legend table, top-right corner at (x1, y0) mm, size x size mm."""
    from matplotlib.patches import Rectangle as Rect
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / Hf)
    has = set(keys)
    rows = [("khdr", None, ""),
            ("head", None, "PDF+PS"),
            ("row", {30: "ref"}, "NSF"), ("row", {30: "gmm"}, "GMM"),
            ("head", None, "Path A (MLP)"),
            ("row", {8: "A_k8"}, "VMIM, 100 h"),
            ("head", None, "Path B (CNN)"),
            *[("row", {4: f"V4_{h}", 8: f"V8_{h}"}, f"VMIM, {h} h") for h in (100, 50)],
            *[("row", {4: f"M4_{h}"}, f"MSE, {h} h") for h in (100, 50)],
            ("truth", None, "true value")]
    rows = [r for r in rows if r[0] != "row" or set(r[1].values()) & has]
    x0 = x1 - size
    rh = size / len(rows)
    lab_w = LEG_LAB_FRAC * size
    kw = (size - lab_w) / 3
    cx = {k: x0 + lab_w + (i + 0.5) * kw for i, k in enumerate((4, 8, 30))}

    def rect(x, y, w, h, fc, ec="none", lw=0.0, z=1.0):
        fig.add_artist(Rect((fx(x), fy(y + h)), w / W, h / Hf, fc=fc, ec=ec, lw=lw, zorder=z,
                            transform=fig.transFigure))

    rect(x0, y0, size, size, pc.WHITE, z=0.9)
    for i, (kind, ks, lab) in enumerate(rows):
        y = y0 + i * rh
        yc = y + 0.5 * rh
        if kind in ("khdr", "head"):
            rect(x0, y, size, rh, LEG_HEAD_FILL)
            if kind == "khdr":
                fig.text(fx(x0 + 1.0), fy(yc), r"$k$", ha="left", va="center_baseline",
                         fontsize=LEG_FS, color=pc.INK, zorder=3)
                for k, c in cx.items():
                    fig.text(fx(c), fy(yc), f"{k}", ha="center", va="center_baseline",
                             fontsize=LEG_FS, color=pc.INK, zorder=3)
            else:
                fig.text(fx(x0 + 1.0), fy(yc), lab, ha="left", va="center_baseline",
                         fontsize=LEG_FS, style="italic", weight="bold", color=pc.INK, zorder=3)
            continue
        fig.text(fx(x0 + (1.0 if kind == "truth" else 2.0)), fy(yc), lab, ha="left",
                 va="center_baseline", fontsize=LEG_FS, color=pc.INK, zorder=3)
        if kind == "truth":
            c = cx[8]
            fig.add_artist(Line2D([fx(c - 0.35 * kw), fx(c + 0.35 * kw)], [fy(yc)] * 2,
                                  transform=fig.transFigure, zorder=3, **TRUTH_LINE))
            fig.add_artist(Line2D([fx(c)], [fy(yc)], ls="none", ms=2.8, zorder=4,
                                  transform=fig.transFigure, **TRUTH_MK))
            continue
        for k, key in ks.items():
            if key not in has:
                continue
            c = cx[k]
            rect(c - 0.5 * kw, y, kw, rh, pc._on_white(al.line_colour(key), LEG_TINT), z=1.5)
            hw = 0.38 * kw
            fig.add_artist(Line2D([fx(c - hw), fx(c + hw)], [fy(yc)] * 2,
                                  color=al.line_colour(key), ls=al.linestyle(key),
                                  lw=al.line_width(key, LW), transform=fig.transFigure, zorder=3))
            mk = al.marker(key)
            ms = {"o": 2.8, "s": 2.6, "^": 3.2, "*": 5.0, "X": 3.8}[mk]
            mkw = (dict(mfc=al.colour(key), mec=pc.INK, mew=0.4) if key not in ("ref", "gmm")
                   else dict(mfc=al.line_colour(key), mec="white", mew=0.4))
            fig.add_artist(Line2D([fx(c)], [fy(yc)], ls="none", marker=mk, ms=ms, zorder=4,
                                  transform=fig.transFigure, **mkw))
    rect(x0, y0, size, size, "none", ec=mpl.colors.to_rgba(pc.INK, LEG_FRAME_ALPHA), lw=aa.LW["axes"], z=5)


def render(S, truth, lims, path):
    d = len(PARAMS)
    T = W - LEFT - RIGHT
    cell = (T - (d - 1) * GAP) / d
    Hf = TOP + T + XLAB + 0.5
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / Hf)

    own = drawn_limits(S)
    nb = [int(min(200, max(BINS, round(BINS * (lims[i][1] - lims[i][0])
                                       / max(own[i][1] - own[i][0], 1e-12))))) for i in range(d)]
    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, Hf * aa.MM))

        def axes(x, y, w, h):
            return fig.add_axes([fx(x), fy(y + h), w / W, h / Hf])
        A = [[None] * d for _ in range(d)]
        for r in range(d):
            for c in range(r + 1):
                ax = axes(LEFT + c * (cell + GAP), TOP + r * (cell + GAP), cell, cell)
                A[r][c] = ax
                jr, jc = PARAMS[r], PARAMS[c]
                if r == c:
                    for k, s in S.items():
                        hist(ax, k, s[:, jr], lims[r], nb[r])
                    if truth is not None:
                        ax.axvline(truth[jr], zorder=100, **TRUTH_LINE)
                    ax.set_xlim(*lims[r]); ax.set_ylim(bottom=0)
                    despine(ax)
                else:
                    for k, s in S.items():
                        contour(ax, k, s[:, jc], s[:, jr], lims[c], lims[r], [nb[c], nb[r]],
                                (SMOOTH * nb[c] / BINS, SMOOTH * nb[r] / BINS))
                    if truth is not None:
                        ax.axvline(truth[jc], zorder=150, **TRUTH_LINE)
                        ax.axhline(truth[jr], zorder=150, **TRUTH_LINE)
                        ax.plot(truth[jc], truth[jr], ms=3.0, zorder=200, **TRUTH_MK)
                    ax.set_xlim(*lims[c]); ax.set_ylim(*lims[r])
                    ticks(ax.yaxis, lims[r])
                ticks(ax.xaxis, lims[c])
                ax.tick_params(labelsize=FS["tick"], pad=aa.AX["tickpad"])
                if r == d - 1:
                    ax.set_xlabel(PLAB[jc], fontsize=FS["label"], labelpad=aa.AX["labelpad"])
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0 and r > 0:
                    ax.set_ylabel(PLAB[jr], fontsize=FS["label"], labelpad=aa.AX["labelpad"])
                elif r != c:
                    ax.tick_params(labelleft=False)
        fig.align_ylabels([A[r][0] for r in range(1, d)])

        table_legend(fig, Hf, W - RIGHT, TOP, LEG_SIZE * (2 * cell + GAP),
                     [k for k in KEYS if k in S])
        fig.savefig(f"{path}.pdf")
        fig.savefig(f"{path}.png", dpi=300)
        plt.close(fig)
    print("wrote", f"{path}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_CORNER)
    a = ap.parse_args()
    aa.use()
    arms = load_arms()
    sims = pick_sims(arms, a.n)
    print("  sims:", sims)
    jobs = [(s, *pooled(arms, s)) for s in sims]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"{NAME}_*"):
        old.unlink()
    for sim, S, truth in jobs:
        render(S, truth, drawn_limits(S), OUT / f"{NAME}_sim{sim}")


if __name__ == "__main__":
    main()
