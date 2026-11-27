#!/usr/bin/env python3
"""I_c (eq. ic) of the learned summaries t against the PDF + PS input statistics, same look as ..."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_summary_information as fsi  # noqa: E402   settings + styles of the I_c figure
aa, pc = fsi.aa, fsi.pc

ROOT = HERE.parent
CACHE = fsi.CACHE_DIR / "t_information.npz"
SEED = 43

FIGURES = {
    "t_ic_patha": ("full", {
        r"Path A, $k=4$":  [("A_k4", 100, "configs_paper/patha/arm_mlp_vmim_t4.yaml")],
        r"Path A, $k=8$":  [("A_k8", 100, "configs_paper/patha/arm_mlp_vmim_t8.yaml")],
        r"Path A, $k=16$": [("A_k16", 100, "configs_paper/patha/arm_mlp_vmim_t16.yaml")],
    }),
    "t_ic_pathb": ("column", {
        "Path B, VMIM": [(f"VMIM_{h}", h, f"configs_paper/pathb/vmim/arm_cnn_vmim_{n}.yaml")
                         for h, n in ((100, "n1"), (50, "nsqrt2"), (25, "n2"))],
        "Path B, MSE":  [(f"MSE_{h}", h, f"configs_paper/pathb/mse/arm_cnn_mse_{n}.yaml")
                         for h, n in ((100, "n1"), (50, "nsqrt2"), (25, "n2"))],
    }),
}
REF_X = {"PDF": 0.0, "PS": 1.25}
T0 = 2.9
REF_MS = 3.6
MODE = "eig"


def summaries_dir(cfg_path):
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    name = cfg["arm_name"].strip()
    if cfg["compressor"].get("tag_arm_with_seed", False):
        name += f"_s{SEED}"
    return Path(cfg["scratch_root"]) / name / "summaries"


def ic_of(d):
    """(per-component I_c, generalised I_c) of summaries/t.npy, grouped by sim."""
    from scipy.linalg import eigh
    t = np.load(d / "t.npy").astype(np.float64)
    _, idx = np.unique(np.load(d / "original_sim_ids.npy"), return_inverse=True)
    n = np.bincount(idx).astype(float)
    m = np.stack([np.bincount(idx, x) / n for x in t.T], 1)
    w = n / n.sum()
    mu = w @ m
    s_b = (m - mu).T @ ((m - mu) * w[:, None])
    s_n = (t.T @ t) / len(t) - (m.T @ (m * w[:, None]))
    per = np.diag(s_b) / np.diag(s_n)
    gen = np.sort(eigh(s_b, s_n, eigvals_only=True))[::-1]
    return per, gen


def compute(recompute=False):
    """{yaml: generalised I_c} (MODE "eig") or {yaml: per-component I_c} (MODE "comp"), both cached in ..."""
    have = dict(np.load(CACHE)) if CACHE.exists() and not recompute else {}
    todo = [y for _, panels in FIGURES.values() for runs in panels.values()
            for *_, y in runs if f"eig|{y}" not in have]
    for y in todo:
        print("  reading", summaries_dir(y), flush=True)
        have[f"comp|{y}"], have[f"eig|{y}"] = ic_of(summaries_dir(y))
    if todo:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **{k: v for k, v in have.items() if "|" in k})
    return {k.split("|", 1)[1]: v for k, v in have.items() if k.startswith(MODE + "|")}


def reference():
    """{(family, z): 5 I_c values} of the PDF + PS inputs at 100 h (fig:summary_ic cache)."""
    snr = np.asarray(np.load(fsi.CACHE)["snr"])
    return {(b, z): snr[off + 5 * fsi.Z_FILE.index(z): off + 5 * fsi.Z_FILE.index(z) + 5]
            for b, off in (("PDF", 0), ("PS", 15)) for z in fsi.Z_PLOT}


def plot(name, width, panels, ic, ref):
    W = aa.WIDTH_MM[width]
    left, right, gap, top, bottom = 11.2, 0.8, 2.0, 4.6, 6.0 if width == "full" else 10.5
    ax_h = 40.0
    H = top + ax_h + bottom
    ks = [len(ic[runs[0][2]]) for runs in panels.values()]
    widths = [T0 + k + 0.2 for k in ks]
    pw = (W - left - right - gap * (len(panels) - 1)) / sum(widths)
    vals = np.concatenate([ic[y] for runs in panels.values() for *_, y in runs]
                          + list(ref.values()))
    lo, hi = 1.0, 10.0 ** np.ceil(np.log10(1.5 * vals.max()))

    def tst(key, h):
        c = pc.ARMS[key][1]
        return dict(color=c, marker=pc.ARMS[key][2], ms=fsi.IC_MS - 0.6, mfc=c,
                    mec=fsi.IC_MARKER_EDGE, mew=0.5, lw=fsi.IC_LW, ls=pc.NOISE_LS[h])

    def rst(b, z):
        c = fsi.Z_COLORS[z]
        return dict(ls="none", marker=fsi.Z_MARKERS[z], ms=REF_MS, mec=fsi.IC_MARKER_EDGE,
                    mew=0.5, mfc=c if b == "PDF" else "white", color=c)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        x0 = left
        for p, ((title, runs), k, wrel) in enumerate(zip(panels.items(), ks, widths)):
            w = wrel * pw
            ax = fig.add_axes([x0 / W, bottom / H, w / W, ax_h / H])
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            e0, e1 = int(np.log10(lo)), int(np.log10(hi))
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(e0, e1 + 1)]))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))
            ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", labelsize=fsi.FS_L["tick"], pad=1.5)
            if p == 0:
                ax.set_ylabel(r"$I_c$", fontsize=fsi.FS_L["label"] + 3, labelpad=1.0)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False)

            for (b, z), v in ref.items():
                ax.plot(np.full(5, REF_X[b] + fsi.DODGE[z]), v, zorder=3, **rst(b, z))
            ax.axvline(T0 - 0.7, color=pc.GUIDE, lw=aa.LW["guide"], ls=aa.DASH, zorder=1)
            xt = T0 + np.arange(k)
            for key, h, y in runs:
                ax.plot(xt, np.sort(ic[y])[::-1], zorder=4, **tst(key, h))

            step = 1 if k <= 8 else 3
            tx = list(REF_X.values()) + list(xt[::step])
            tl = list(REF_X) + [str(i + 1) for i in range(0, k, step)]
            ax.set_xticks(tx, tl)
            ax.xaxis.set_minor_locator(FixedLocator(xt))
            ax.set_xlim(-0.55, T0 + k - 0.4)
            ax.tick_params(axis="x", labelsize=fsi.FS_L["tick"], pad=2.0)
            ax.text(0.0, 1.03, rf"$\mathbf{{({'abcdefgh'[p]})}}$ {title}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=fsi.FS_L["family"] - 1, color=fsi.IC_TEXT_COLOR, zorder=6)
            x0 += w + gap

        hs = [Line2D([], [], **rst("PDF", z)) for z in fsi.Z_PLOT]
        labs = [rf"$z={z}$" for z in fsi.Z_PLOT]
        hs += [Line2D([], [], **rst("PS", fsi.Z_PLOT[1]))]
        labs += ["open = PS"]
        seen = []
        for runs in panels.values():
            for key, h, _ in runs:
                if (h not in seen) and len(runs) > 1:
                    seen.append(h)
                    hs.append(Line2D([], [], color=pc.INK, lw=fsi.IC_LW, ls=pc.NOISE_LS[h]))
                    labs.append(f"{h} h")
        if width == "full":
            lg_kw = dict(loc="upper right", bbox_to_anchor=((W - right - 1.0) / W,
                                                            1 - (top + 1.0) / H), ncol=2)
        else:
            lg_kw = dict(loc="lower center", bbox_to_anchor=((left + W - right) / 2 / W,
                                                             0.3 / H), ncol=4)
        lg = fig.legend(hs, labs, **lg_kw,
                        fontsize=fsi.FS_L["legend"] - 1.5, frameon=True, fancybox=False,
                        framealpha=1.0, edgecolor=fsi.IC_LEGEND_FRAME,
                        facecolor=fsi.IC_LEGEND_FACE, borderpad=0.3, labelspacing=0.2,
                        handlelength=1.8, columnspacing=1.0, handletextpad=0.4,
                        borderaxespad=0.0)
        lg.get_frame().set_linewidth(aa.LW["axes"])
        fsi.save(fig, name)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="re-read summaries/t.npy")
    a = ap.parse_args()
    ic = compute(a.recompute)
    ref = reference()
    r = np.concatenate(list(ref.values()))
    print(f"  reference PDF+PS (100 h): I_c {r.min():.0f} - {r.max():.0f}, median {np.median(r):.0f}")
    for name, (width, panels) in FIGURES.items():
        for title, runs in panels.items():
            for key, h, y in runs:
                v = ic[y]
                print(f"  {title:16s} {h:3d} h  I_c min {v.min():6.0f}  median "
                      f"{np.median(v):6.0f}  max {v.max():6.0f}")
        plot(name, width, panels, ic, ref)


if __name__ == "__main__":
    main()
