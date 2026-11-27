#!/usr/bin/env python3
"""Paper Fig. 1 (fig:cube_slices): one Loreli II simulation, three redshifts, three noise levels."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

import cmasher as cmr  # noqa: E402,F401     pip install cmasher; registers "cmr.<name>"
try:
    import cmastro as cma  # noqa: E402,F401        # "cma:<name>". Its import fails on
except Exception:
    cmastro = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, mm layout, save

import paper_colors as pc  # noqa: E402
INK = pc.INK
GUIDE = pc.GUIDE
BOX_FACE = pc.WHITE
TICK_COLOR = pc.INK
TEXT_ON_DARK = pc.WHITE
TEXT_SWITCH = 0.45

PANEL_LABEL_FS = {"half": 7.5, "full": 8.5}
PANEL_NOTE_FS = {"half": 6.0, "full": 7.5}

CUBE_CMAP = pc.DIV_NAME
CUBE_CMAP_RANGE = (0.0, 1.0)

IMAGES = HERE.parent / "images"


def cube_cmap():
    """CUBE_CMAP, cut to CUBE_CMAP_RANGE."""
    lo, hi = CUBE_CMAP_RANGE
    if (lo, hi) == (0.0, 1.0):
        return mpl.colormaps[CUBE_CMAP]
    return cmr.get_sub_cmap(CUBE_CMAP, lo, hi)


DATA = Path("/data/ddehiwalage-don/data/dtb_data")
SIMIDS = Path("/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy")
Z = ["12.06", "10.32", "8.18"]
N, L_BOX = 32, 200.0
REF_HOURS = 100.0

FS = {"label": aa.AX["label"], "tick": aa.AX["tick"], "annot": aa.FS["legend"],
      "legend": aa.FS["legend"], "note": aa.FS["note"]}

HIST_STYLE = {
    "clean":        pc.VOXEL_PDF["clean"],
    "mean removed": pc.VOXEL_PDF["mean removed"],
    "noise":        [pc.VOXEL_PDF[h] for h in pc.NOISE_HOURS],
}


def remove_mean(cube):
    """Subtract the whole-cube mean (one number per cube)."""
    return cube - cube.mean()


def load_cubes(sim_idx, sim_id, noise_idx, n_levels, independent):
    """Clean cubes and 100 h noise cubes (float64) for the three redshifts."""
    if sim_id is None:
        sim_id = int(np.load(SIMIDS)[sim_idx])
    clean, noise = [], [[] for _ in range(n_levels)]
    for z in Z:
        c = np.memmap(DATA / f"clean_cubes_z={z}.dat", np.float64, "r").reshape(-1, N, N, N)
        n = np.memmap(DATA / f"noise_cubes_100h_z={z}.dat", np.float64, "r").reshape(-1, N, N, N)
        clean.append(np.asarray(c[sim_id], np.float64))
        for i in range(n_levels):
            j = (noise_idx + (i if independent else 0)) % n.shape[0]
            noise[i].append(np.asarray(n[j], np.float64))
    return clean, noise, sim_id


def demo_cubes(n_levels, independent, seed=0):
    """Synthetic stand-in (bubbly, z-dependent) for testing the layout only."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N) * N
    K = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    K[0, 0, 0] = 1
    base = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N, N, N))) * K ** -1.4).real
    base /= base.std()
    clean, noise = [], [[] for _ in range(n_levels)]
    for mean, amp, nsig in [(-58, 8.6, 11.0), (-112, 13.2, 10.9), (-147, 10.2, 11.0)]:
        clean.append(mean + amp * base)
        draw = nsig * rng.standard_normal((N, N, N))
        for i in range(n_levels):
            noise[i].append(nsig * rng.standard_normal((N, N, N)) if independent and i else draw)
    return clean, noise, -1


def nice_ticks(vmax, lin):
    """Symmetric ticks for the symlog colour bar: two in the linear part, then 50/100/200/..."""
    step = 5 if lin <= 12 else 10
    linear = [v for v in np.arange(step, lin + 0.1, step)][-2:]
    logp = [v for v in (50, 100, 200, 500) if 1.6 * lin < v <= vmax * 1.001]
    keep = [float(v) for v in linear] + [float(v) for v in logp]
    return sorted([-v for v in keep] + [0.0] + keep)


def hours_label(h):
    return f"{h:.0f} h" if h >= 10 else f"{h:.1f} h"


def plot(clean, noise100, hours, sl, fname, norm_mode, cmap, hist, y_range=L_BOX):
    nz, nlev = len(Z), len(hours)
    scale = [np.sqrt(REF_HOURS / h) for h in hours]
    means = [c.mean() for c in clean]
    clean_rm = [remove_mean(c) for c in clean]
    noisy_rm = [[remove_mean(clean[iz] + s * noise100[il][iz]) for iz in range(nz)]
                for il, s in enumerate(scale)]
    sig_s = [c.std() for c in clean]
    sig_n = [[s * noise100[il][iz].std() for iz in range(nz)] for il, s in enumerate(scale)]
    ic = [[(sig_s[iz] / sig_n[il][iz]) ** 2 for iz in range(nz)] for il in range(nlev)]

    ny = int(round(N * y_range / L_BOX))
    y_range = ny * L_BOX / N
    imgs = [[im[:, :ny] for im in
             [clean[iz][sl], clean_rm[iz][sl]] + [noisy_rm[il][iz][sl] for il in range(nlev)]]
            for iz in range(nz)]
    ncol = 2 + nlev

    allv = np.abs(np.stack([im for row in imgs for im in row]))
    vmax = float(np.ceil(np.percentile(allv, 99.7) / 10) * 10)
    lin = float(np.ceil(np.percentile(np.abs(np.stack([r[1] for r in imgs])), 99.5) / 5) * 5)
    if norm_mode == "symlog":
        norm = SymLogNorm(linthresh=lin, linscale=1.2, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = Normalize(-vmax, vmax)

    W = aa.WIDTH_MM["full"]
    left, gap, vgap = 11.6, 4.2, 2.8
    cb_gap, cb_w, cb_lab = 1.6, 2.4, 11.0
    hist_gap, hist_rel, right = 4.5, 1.25, 1.0
    top = 6.6 if hist else 0.8
    bottom = 8.5
    n_units = ncol + (hist_rel if hist else 0)
    pw = (W - left - (ncol - 1) * gap - cb_gap - cb_w - cb_lab
          - (hist_gap if hist else 0) - right) / n_units
    ph = pw * y_range / L_BOX
    H = top + nz * ph + (nz - 1) * vgap + bottom
    x_img = [left + c * (pw + gap) for c in range(ncol)]
    x_cb = x_img[-1] + pw + cb_gap
    x_hist = x_cb + cb_w + cb_lab + hist_gap
    hw = hist_rel * pw

    def y_row(r):
        return top + r * (ph + vgap)

    def axes(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    labels = (["clean", r"$\mathbf{\langle\delta T_b\rangle}$ removed"]
              + [f"+ {hours_label(h)} noise" for h in hours])

    def text_color(img, x, y, ha, va):
        if img is None:
            return INK
        ny_, nx_ = img.T.shape
        xs = slice(0, int(0.45 * nx_)) if ha == "left" else slice(int(0.55 * nx_), nx_)
        ys = slice(int(0.7 * ny_), ny_) if va == "top" else slice(0, int(0.3 * ny_))
        rgb = cmap(norm(img.T[ys, xs]))[..., :3]
        lum = float((rgb @ [0.299, 0.587, 0.114]).mean())
        return TEXT_ON_DARK if lum < TEXT_SWITCH else INK

    shape = "full" if y_range > 150 else "half"
    fs_label, fs_note = PANEL_LABEL_FS[shape], PANEL_NOTE_FS[shape]
    if shape == "full":
        labels = [lab.replace(" removed", "\nremoved").replace(" noise", "\nnoise")
                  for lab in labels]

    def panel_text(ax, x, y, txt, ha, va, fs, img=None):
        ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
                weight="bold", color=text_color(img, x, y, ha, va), zorder=8)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        im = None
        for r in range(nz):
            for c in range(ncol):
                ax = axes(x_img[c], y_row(r), pw, ph)
                im = ax.imshow(imgs[r][c].T, origin="lower", extent=(0, L_BOX, 0, y_range),
                               cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
                last, first = r == nz - 1, c == 0
                ax.set_xticks([0, 100, 200], ["0", "100", "200"])
                yt = [0, 100, 200] if y_range > 150 else [0, 50, 100] if y_range > 60 else [0, 25, 50]
                yt = [v for v in yt if v <= y_range + 1e-6]
                ax.set_yticks(yt, [f"{v:g}" for v in yt])
                aa.minor_ticks(ax, x=25, y=25)
                ax.tick_params(which="both", color=TICK_COLOR)
                ax.tick_params(which="major", labelsize=FS["tick"], labelbottom=last,
                               labelleft=first)
                if c == 0:
                    txt = rf"$\mathbf{{\langle\delta T_b\rangle={means[r]:.0f}}}$ mK"
                elif c == 1:
                    txt = rf"$\mathbf{{\sigma={sig_s[r]:.1f}}}$ mK"
                else:
                    il = c - 2
                    txt = rf"$\mathbf{{\sigma/\sigma_n={sig_s[r] / sig_n[il][r]:.2f}}}$"
                panel_text(ax, 0.97, 0.05, txt, "right", "bottom", fs_note, imgs[r][c])
                if r == 0:
                    panel_text(ax, 0.035, 0.955, f"({'abcdefgh'[c]}) {labels[c]}", "left",
                               "top", fs_label, imgs[r][c])
            ax0 = fig.axes[-ncol]
            yz = 0.955
            ax0.text(0.965, yz, rf"$z={Z[r]}$", transform=ax0.transAxes, ha="right",
                     va="top", fontsize=fs_note + 0.5, zorder=8,
                     color=text_color(imgs[r][0], 0.965, yz, "right", "top"))

        fig.text((x_img[0] + x_img[-1] + pw) / 2 / W, 1 - (H - 1.0) / H,
                 r"$x$ ($h^{-1}\,{\rm cMpc}$)", ha="center", va="bottom",
                 fontsize=FS["label"], color=INK)
        fig.text(2.4 / W, 1 - (y_row(0) + (nz * ph + (nz - 1) * vgap) / 2) / H,
                 r"$y$ ($h^{-1}\,{\rm cMpc}$)", rotation=90, ha="center", va="center",
                 fontsize=FS["label"], color=INK)

        cax = axes(x_cb, y_row(0), cb_w, nz * ph + (nz - 1) * vgap)
        cb = fig.colorbar(im, cax=cax)
        if norm_mode == "symlog":
            t = nice_ticks(vmax, lin)
            cb.set_ticks(t, labels=[f"${v:g}$" for v in t])
        cb.minorticks_off()
        cb.outline.set_linewidth(aa.LW["axes"])
        cb.ax.tick_params(labelsize=FS["tick"], length=2.2, width=aa.LW["axes"], pad=aa.AX["tickpad"])
        cb.set_label(r"$\delta T_b$ (mK)", fontsize=FS["label"], labelpad=aa.AX["labelpad"])

        if hist:
            lo = min(np.percentile(c, 0.1) for c in clean)
            hi = max(np.percentile(x, 99.9) for lev in noisy_rm for x in lev)
            lo, hi = np.floor(lo / 10) * 10 - 10, np.ceil(hi / 10) * 10 + 5
            bins = np.linspace(lo, hi, 120)
            for r in range(nz):
                ax = axes(x_hist, y_row(r), hw, ph)
                curves = [("clean", clean[r], HIST_STYLE["clean"]),
                          (r"$\langle\delta T_b\rangle$ removed", clean_rm[r], HIST_STYLE["mean removed"])]
                curves += [(hours_label(h), noisy_rm[il][r], HIST_STYLE["noise"][il % 3])
                           for il, h in enumerate(hours)]
                ymax = 0
                for lab, cube, st in curves:
                    h_, e = np.histogram(cube.ravel(), bins=bins, density=True)
                    ax.step(e[:-1], h_, where="post", label=lab, zorder=3, **st)
                    ymax = max(ymax, h_.max())
                ax.axvline(0, color=GUIDE, lw=aa.LW["cross"], zorder=1)
                ax.set_xlim(lo, hi)
                ax.set_ylim(0, ymax * 1.38)
                ax.set_yticks([])
                ax.tick_params(axis="x", which="both", top=False)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.xaxis.set_minor_locator(MultipleLocator(25))
                ax.tick_params(axis="x", which="major", length=2.6, pad=1.5,
                               labelsize=FS["tick"], labelbottom=(r == nz - 1))
                ax.tick_params(axis="x", which="minor", length=1.5)
                if r == 0:
                    panel_text(ax, 0.035, 0.955, f"({'abcdefgh'[ncol]}) voxel PDF", "left",
                               "top", fs_label)
                if r == nz - 1:
                    ax.set_xlabel(r"$\delta T_b$ (mK)", fontsize=FS["label"], labelpad=1.5)
                if r == 0:
                    hs, ls = ax.get_legend_handles_labels()
                    fig.legend(hs, ls, frameon=False, fontsize=FS["legend"],
                               loc="upper right", ncol=3,
                               bbox_to_anchor=((W - right) / W, 1 - 0.2 / H),
                               handlelength=1.8, borderaxespad=0.0, labelspacing=0.15,
                               handletextpad=0.4, columnspacing=1.0)

        aa.save(fig, fname, outdir=IMAGES)
        plt.close(fig)

    print("  <dTb> = " + ", ".join(f"{m:.1f}" for m in means) +
          " mK; sigma = " + ", ".join(f"{s:.1f}" for s in sig_s) + " mK")
    for il, h in enumerate(hours):
        print(f"  {hours_label(h):>6s}: sigma_n = " + ", ".join(f"{s:.1f}" for s in sig_n[il]) +
              " mK, sigma/sigma_n = " + ", ".join(f"{np.sqrt(s):.2f}" for s in ic[il]) +
              ", I_c(vox) = " + ", ".join(f"{s:.2f}" for s in ic[il]))
    if norm_mode == "symlog":
        print(f"  symlog linthresh = {lin:g} mK, vmax = {vmax:g} mK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-idx", type=int, default=0, help="index in the masked sim list")
    ap.add_argument("--sim-id", type=int, default=None, help="file row (overrides --sim-idx)")
    ap.add_argument("--noise-idx", type=int, default=0, help="noise-bank realisation")
    ap.add_argument("--hours", type=float, nargs="+", default=[100, 50, 25],
                    help="integration times of the noise columns (default 100 50 25)")
    ap.add_argument("--independent-noise", action="store_true",
                    help="a different noise cube for each level (default: the same one, scaled)")
    ap.add_argument("--slice", type=int, default=N // 2, help="sky-plane index along axis 0")
    ap.add_argument("--norm", choices=["symlog", "linear"], default="symlog")
    ap.add_argument("--hist", action=argparse.BooleanOptionalAction, default=False,
                    help="voxel-PDF column (default off, as in the paper; --hist)")
    ap.add_argument("--y-range", type=float, default=100.0,
                    help="h^-1 cMpc shown along y (default 100 = half the box): wide, "
                         "short panels with square voxels; 200 = full square slices")
    ap.add_argument("--name", default=None, help="output name (default fig_1_cube_slices)")
    ap.add_argument("--demo", action="store_true", help="synthetic data (layout test)")
    a = ap.parse_args()
    aa.use()

    n = len(a.hours)
    if a.demo:
        clean, noise, sid = demo_cubes(n, a.independent_noise)
    else:
        clean, noise, sid = load_cubes(a.sim_idx, a.sim_id, a.noise_idx, n, a.independent_noise)
    name = a.name or "fig_1_cube_slices" + ("_demo" if a.demo else "")
    print(f"sim file row {sid}, noise {a.noise_idx}"
          f"{' (+1, +2 ...)' if a.independent_noise else ' (scaled)'}, slice {a.slice}, "
          f"hours {a.hours}")
    plot(clean, noise, a.hours, a.slice, name, a.norm, cube_cmap(), a.hist,
         y_range=a.y_range)


if __name__ == "__main__":
    main()
