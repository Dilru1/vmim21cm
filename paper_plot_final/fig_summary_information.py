#!/usr/bin/env python3
"""The two summary-statistic figures of Sect."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

import cmasher as cmr  # noqa: E402           pip install cmasher; registers "cmr.<name>"
try:
    import cmastro  # noqa: E402,F401        # "cma:<name>". Its import fails on
except Exception:
    cmastro = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sty"))
import aa_style as aa  # noqa: E402            fonts, lines, ticks, mm layout, save
import aa_color_style as ac  # noqa: E402      colour helpers (darken, lighten, text_on)

import paper_colors as pc  # noqa: E402
INK = pc.INK
WHITE = pc.WHITE

Z_COLORS = dict(pc.Z_COLORS)
Z_MARKERS = dict(pc.Z_MARKERS)
IC_MARKER_EDGE = INK
IC_REF_COLOR = pc.INK
IC_TEXT_COLOR = INK
IC_LEGEND_FRAME = INK
IC_LEGEND_FACE = WHITE
VOX_TINT_BY_HOURS = [0.15, 0.45, 0.72]
VOX_EDGE_DARKEN = 0.3
VOX_MEDIAN_COLOR = INK

HM_CMAP = pc.SEQ_CMAP
HM_GRID_COLOR = WHITE
HM_FRAME_COLOR = INK
HM_TEXT_DARK = INK
HM_TEXT_LIGHT = WHITE
HATCH_FACE = pc.HATCH_FACE
HATCH_COLOR = pc.HATCH_COLOR

FAMILY_COLORS = {"PDF": pc.INK, "PS": pc.MUTED, "mean": pc.GUIDE}
FAMILY_TEXT_DARKEN = 0.25
DOTS_Z_COLORS = dict(pc.Z_COLORS)
OLD_HEATMAP_CMAP = pc.SEQ_CMAP
OLD_HEATMAP_TEXT_SWITCH = 0.55

FS = {"label": aa.FS["label"], "tick": aa.FS["tick"], "annot": aa.FS["legend"],
      "tiny": aa.FS["note"] - 0.5}
LW_REF = aa.LW["guide"] + 0.2
LW_THIN = aa.LW["cross"]
PARAM_LABELS = {"fx": r"$\log_{10} f_{\rm X}$", "tau": r"$\tau$", "rhs": r"$r_{\rm H/S}$",
                "mmin": r"$\log_{10} M_{\rm min}$", "fesc": r"$f_{\rm esc}$"}
PARAM_ORDER = ["fx", "tau", "rhs", "mmin", "fesc"]
IMAGES = HERE / "images"
COLUMN_MM = aa.WIDTH_MM["column"]


def save(fig, name):
    """images/<name>.pdf + .svg at the drawn size (aa_style)."""
    aa.save(fig, name, outdir=IMAGES)


CACHE_DIR = HERE.parent / "paper_plots" / "cache"
CACHE = CACHE_DIR / "summary_information.npz"
Z_FILE = ["8.18", "10.32", "12.06"]
Z_PLOT = ["12.06", "10.32", "8.18"]
PDF_NAMES = [r"$\ell_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_5$", r"$\tau_6$"]
PS_NAMES = [rf"$k_{i}$" for i in range(1, 6)]
PARAMS = [PARAM_LABELS[k] for k in PARAM_ORDER]
COL = FAMILY_COLORS
FAMILY = {"PDF": "PDF", "PS": "PS", "mean": r"$\langle\delta T_b\rangle$", "vox": "voxel"}
CLIP = 3e4


def zlab(z, eq=True):
    """Bold black redshift label: z = 12.06 (z bold italic, digits bold upright)."""
    return rf"$\mathbfit{{z}}=\mathbf{{{z}}}$" if eq else rf"$\mathbf{{{z}}}$"


def family_text(b):
    return dict(fontsize=FS["tick"], weight="bold", color=ac.darken(COL[b], FAMILY_TEXT_DARKEN))


def column_layout():
    """[(npz column, label, block, z)] in plotted order: PDF, PS, mean."""
    cols = []
    for b, names, off in (("PDF", PDF_NAMES, 0), ("PS", PS_NAMES, 15)):
        for zp in Z_PLOT:
            zi = Z_FILE.index(zp)
            cols += [(off + 5 * zi + i, names[i], b, zp) for i in range(5)]
    for zp in Z_PLOT:
        cols.append((30 + Z_FILE.index(zp), zp, "mean", zp))
    return cols


def load(cache):
    c = np.load(cache)
    cols = column_layout()
    idx = [k[0] for k in cols]
    rI = np.sqrt(1.0 - np.exp(-2.0 * np.clip(c["mi"][:, idx], 0, None)))
    return rI, np.asarray(c["snr"])[idx], int(c["n_used"]), cols


def groups(cols):
    """[(block, z, first, last)] runs in plotted order; <dTb> is one group."""
    out = []
    for i, c in enumerate(cols):
        key = (c[2], c[3] if c[2] != "mean" else "")
        if not out or tuple(out[-1][:2]) != key:
            out.append([key[0], key[1], i, i])
        out[-1][3] = i
    return out


FS_L = {"label": 9.0, "tick": 8.0, "legend": 8.5, "family": 9.0}
MS_L, LW_L = 4.6, 0.9
DODGE = dict(zip(Z_PLOT, [-0.2, 0.0, 0.2]))


def _zstyle(z):
    c = Z_COLORS[z]
    return dict(color=c, marker=Z_MARKERS[z], ms=MS_L, mfc=c,
                mec=IC_MARKER_EDGE, mew=aa.LW["edge"], lw=LW_L)


def _z_legend(fig, W, H, x0_mm, x1_mm, y_top_mm):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], **_zstyle(z)) for z in Z_PLOT]
    fig.legend(hs, [zlab(z) for z in Z_PLOT], loc="upper center",
               bbox_to_anchor=((x0_mm + x1_mm) / 2 / W, 1 - y_top_mm / H), ncol=3,
               fontsize=FS_L["legend"], handlelength=2.0, columnspacing=1.8,
               handletextpad=0.5, borderaxespad=0.0)


def _family_title(fig, W, H, x_mid_mm, y_mm, b):
    fig.text(x_mid_mm / W, 1 - y_mm / H, FAMILY[b], ha="center", va="baseline",
             fontsize=FS_L["family"], weight="bold", color=INK)


def _xstats(ax, names, show):
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xticks(range(len(names)), names if show else [""] * len(names),
                  fontsize=FS_L["tick"])
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="x", which="major", length=0.9 * aa.TICK["major"], pad=2.0)


DATA = Path("/data/ddehiwalage-don/data/dtb_data")
SIMIDS = Path("/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy")
NPZ = Path("/gscratch/ddehiwalage-don/sbi_runs/BASELINE/baseline_pdf_ps_mean/"
           "baseline_pdf_ps_mean_Nnoise1000.npz")
VOX_CACHE = CACHE_DIR / "vox_ic.npz"
VOX_MI_CACHE = CACHE_DIR / "vox_mi.npz"
N_VOX = 32
N_NOISE = 1000
VOX_HOURS = [100.0, 50.0, 25.0]
VOX_TINT = dict(zip(VOX_HOURS, VOX_TINT_BY_HOURS))


def compute_vox_ic(n_sims=9120, seed=0, cache=VOX_CACHE, recompute=False):
    """Voxel variances at 100 h for n_sims random simulations (all 9120 by default), each with one random ..."""
    cache = Path(cache)
    if cache.exists() and not recompute:
        c = np.load(cache)
        if "vc12.06" in c and int(c["n_sims"]) == n_sims and int(c["seed"]) == seed:
            print(f"loaded {cache}")
            return {z: (c[f"vc{z}"], c[f"vn{z}"]) for z in Z_PLOT}
    rng = np.random.default_rng(seed)
    ids = np.load(SIMIDS)
    n_sims = min(n_sims, len(ids))
    pick = np.sort(rng.choice(len(ids), n_sims, replace=False))
    out = {}
    for z in Z_PLOT:
        cl = np.memmap(DATA / f"clean_cubes_z={z}.dat", np.float64, "r").reshape(-1, N_VOX, N_VOX, N_VOX)
        nb = np.memmap(DATA / f"noise_cubes_100h_z={z}.dat", np.float64, "r").reshape(-1, N_VOX, N_VOX, N_VOX)
        nidx = rng.integers(0, nb.shape[0], n_sims)
        vn_bank = np.array([np.var(np.asarray(nb[j])) for j in range(nb.shape[0])])
        vc = np.array([np.var(np.asarray(cl[int(ids[i])])) for i in pick])
        out[z] = (vc, vn_bank[nidx])
        print(f"  z={z}: {n_sims} sims read")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, n_sims=n_sims, seed=seed, sim_rows=ids[pick],
             **{f"vc{z}": out[z][0] for z in Z_PLOT}, **{f"vn{z}": out[z][1] for z in Z_PLOT})
    return out


def compute_vox_mi(n_pos=20, k=5, seed=0, cache=VOX_MI_CACHE, recompute=False):
    """r_I between ONE voxel and each parameter, with the same kNN estimator as the statistics."""
    cache = Path(cache)
    keys = ("r", "sd", "null", "clean")
    if cache.exists() and not recompute:
        c = np.load(cache)
        if int(c["n_pos"]) == n_pos and int(c["seed"]) == seed:
            print(f"loaded {cache}")
            return {z: {q: c[f"{q}{z}"] for q in keys} for z in Z_PLOT}
    from sklearn.feature_selection import mutual_info_regression
    theta = np.load(NPZ)["theta"][::N_NOISE].astype(np.float64)
    ids = np.load(SIMIDS)
    assert len(ids) == len(theta), (f"{len(ids)} cube ids but {len(theta)} simulations "
                                    "in the npz: the simulation order must match")
    n, npar = theta.shape
    rng = np.random.default_rng(seed)
    pos = tuple(rng.integers(0, N_VOX, (3, n_pos)))

    def r_of(u, t):
        I = mutual_info_regression(u[:, None], t, n_neighbors=k, random_state=seed)[0]
        return np.sqrt(1.0 - np.exp(-2.0 * max(I, 0.0)))

    out = {}
    for z in Z_PLOT:
        cl = np.memmap(DATA / f"clean_cubes_z={z}.dat", np.float64, "r").reshape(-1, N_VOX, N_VOX, N_VOX)
        nb = np.memmap(DATA / f"noise_cubes_100h_z={z}.dat", np.float64, "r").reshape(-1, N_VOX, N_VOX, N_VOX)
        nidx = rng.integers(0, nb.shape[0], n)
        uc, un = np.empty((n, n_pos)), np.empty((n, n_pos))
        for a, row in enumerate(ids):
            cube = np.asarray(cl[int(row)])
            uc[a] = cube[pos] - cube.mean()
            un[a] = np.asarray(nb[nidx[a]])[pos]
        r = np.zeros((len(VOX_HOURS), n_pos, npar))
        r0 = np.zeros_like(r)
        for ih, h in enumerate(VOX_HOURS):
            u = uc + np.sqrt(100.0 / h) * un
            for q in range(n_pos):
                for j in range(npar):
                    r[ih, q, j] = r_of(u[:, q], theta[:, j])
                    r0[ih, q, j] = r_of(u[:, q], theta[rng.permutation(n), j])
        clean = np.array([[r_of(uc[:, q], theta[:, j]) for j in range(npar)]
                          for q in range(n_pos)]).mean(0)
        out[z] = dict(r=r.mean(1), sd=r.std(1), null=r0.mean(1), clean=clean)
        print(f"  z={z}: voxel r_I 100/50/25 h {np.round(out[z]['r'], 2).tolist()}")
        print(f"         floor (shuffled)    {np.round(out[z]['null'], 2).tolist()}")
        print(f"         clean voxel         {np.round(clean, 2).tolist()}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, n_pos=n_pos, seed=seed,
             **{f"{q}{z}": out[z][q] for z in Z_PLOT for q in keys})
    return out


def vox_stats(vox, hours):
    """Per (z, t): box statistics of the per-simulation I_c(vox) and the ensemble value."""
    st = {}
    for z in Z_PLOT:
        vc, vn = (np.asarray(a, float) for a in vox[z])
        for h in hours:
            f = 100.0 / h
            ic = vc / (vn * f)
            q = np.percentile(ic, [5, 25, 50, 75, 95])
            st[(z, h)] = dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4],
                              fliers=[], ens=vc.mean() / (vn.mean() * f), n=ic.size)
    return st


def _vox_panel(ax, vox, hours=VOX_HOURS):
    st = vox_stats(vox, hours)
    for ih, h in enumerate(hours):
        for z in Z_PLOT:
            c = Z_COLORS[z]
            x0 = ih + 1.3 * DODGE[z]
            s = st[(z, h)]
            ax.bxp([dict(s, label="")], positions=[x0], widths=0.24, showfliers=False,
                   patch_artist=True, manage_ticks=False, zorder=2,
                   boxprops=dict(facecolor=ac.lighten(c, VOX_TINT.get(h, 0.45)),
                                 edgecolor=ac.darken(c, VOX_EDGE_DARKEN), linewidth=0.6),
                   medianprops=dict(color=VOX_MEDIAN_COLOR, linewidth=1.0),
                   whiskerprops=dict(color=ac.darken(c, VOX_EDGE_DARKEN), linewidth=0.6),
                   capprops=dict(color=ac.darken(c, VOX_EDGE_DARKEN), linewidth=0.6))
            ax.plot([x0], [s["ens"]], ls="none", marker=Z_MARKERS[z], ms=2.9,
                    mfc=WHITE, mec=IC_MARKER_EDGE, mew=0.8, zorder=4)
    ax.set_xlim(-0.55, len(hours) - 0.45)
    ax.set_xticks(range(len(hours)), [f"{h:g}" for h in hours], fontsize=FS_L["tick"])
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="x", which="major", length=0.9 * aa.TICK["major"], pad=2.0)
    ax.set_xlabel("noise [h]", fontsize=FS_L["tick"], labelpad=0.5)
    return st


IC_LEGEND_IN = ("PDF", "PS")
IC_MS, IC_LW = 5.4, 1.3


def plot_ic_large(Ic, cols, name="summary_ic_column", vox=None, legend_in=IC_LEGEND_IN):
    """I_c of each statistic, (a) PDF | (b) PS (| (c) voxel), one log axis."""
    Ic = np.asarray(Ic)
    W = COLUMN_MM
    left, right, gap, top, bottom = 11.2, 0.8, 1.8, 0.9, 6.3
    ax_h = 35.4
    H = top + ax_h + bottom
    panels = [("PDF", 1.0), ("PS", 1.0)] + ([("vox", 0.95)] if vox else [])
    pw = (W - left - right - (len(panels) - 1) * gap) / sum(w for _, w in panels)
    keep = [i for i, c in enumerate(cols) if c[2] != "mean"]
    lo = 0.5 if not vox else 1e-2
    hi = 10.0 ** np.ceil(np.log10(1.5 * Ic[keep].max()))
    f1 = (0 - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    vst = None

    def zst(z):
        st = _zstyle(z)
        st.update(ms=IC_MS, lw=IC_LW, mew=0.6, mec=IC_MARKER_EDGE)
        return st

    def legend(ax, b):
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], **zst(z)) for z in Z_PLOT]
        kw = dict(loc="upper right", bbox_to_anchor=(0.975, 0.975))
        if b == "PS":
            kw = dict(loc="lower left", bbox_to_anchor=(0.025, f1 + 0.025))
        lg = ax.legend(hs, [rf"$z={z}$" for z in Z_PLOT], fontsize=FS_L["legend"] - 0.5,
                       frameon=True, fancybox=False, framealpha=1.0,
                       edgecolor=IC_LEGEND_FRAME, facecolor=IC_LEGEND_FACE,
                       borderpad=0.35, labelspacing=0.25, handlelength=2.0,
                       handletextpad=0.5, borderaxespad=0.0, **kw)
        lg.get_frame().set_linewidth(aa.LW["axes"])
        lg.set_zorder(5)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        x0 = left
        for p, (b, wrel) in enumerate(panels):
            w = wrel * pw
            ax = fig.add_axes([x0 / W, bottom / H, w / W, ax_h / H])
            ax.axhline(1.0, color=IC_REF_COLOR, lw=1.0, ls=aa.DASH, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            e0, e1 = int(np.ceil(np.log10(lo))), int(np.log10(hi))
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(e0, e1 + 1)]))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))
            ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="y", labelsize=FS_L["tick"], pad=1.5)
            if p == 0:
                ax.set_ylabel(r"$I_c$", fontsize=FS_L["label"] + 3, labelpad=1.0)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False)

            if b == "vox":
                vst = _vox_panel(ax, vox)
                title = "voxel"
            else:
                for z in Z_PLOT:
                    idx = [i for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                    ax.plot(np.arange(5), Ic[idx], zorder=3, **zst(z))
                _xstats(ax, PDF_NAMES if b == "PDF" else PS_NAMES, True)
                title = FAMILY[b]
            ax.text(0.035, 0.965, rf"$\mathbf{{({'abc'[p]})}}$ {title}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=FS_L["family"], color=IC_TEXT_COLOR, zorder=6)
            if b in legend_in:
                legend(ax, b)
            if p == len(panels) - 1:
                ax.text(0.97, f1 + 0.02, r"$I_c=1$", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=FS_L["tick"], color=IC_TEXT_COLOR, zorder=6)
            x0 += w + gap

        save(fig, name)
        plt.close(fig)
    if vst:
        print("  I_c(vox): ensemble value [median, 25-75 %] over "
              f"{vst[(Z_PLOT[0], VOX_HOURS[0])]['n']} simulations")
        for h in VOX_HOURS:
            print(f"    {h:5g} h: " + "; ".join(
                f"z={z} {vst[(z, h)]['ens']:.2f} [{vst[(z, h)]['med']:.2f}, "
                f"{vst[(z, h)]['q1']:.2f}-{vst[(z, h)]['q3']:.2f}]" for z in Z_PLOT))
    return W, H


def _text_color(v):
    return ac.text_on(HM_CMAP(float(v)), dark=HM_TEXT_DARK, light=HM_TEXT_LIGHT)


HATCH = "////"
HATCH_RC = {"hatch.linewidth": 0.45, "hatch.color": HATCH_COLOR}
CB_LABEL = r"$r_I=\sqrt{1-e^{-2I}}$"
FLOOR_TEXT = r"$r_I\approx 0$ (below the estimator floor)"


def _family_rI(rI, cols, j, b):
    return np.array([[rI[j, i] for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                     for z in Z_PLOT]).T


def _hm_block(fig, W, H, x0, y0, w, h, M, floor, subs=None, fs=6.6):
    n, m = M.shape
    ax = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])
    ax.imshow(np.ma.masked_less(M, floor), aspect="auto", cmap=HM_CMAP, vmin=0, vmax=1,
              extent=(-0.5, m - 0.5, n - 0.5, -0.5), interpolation="nearest", zorder=1)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i in range(n):
        for k in range(m):
            v = M[i, k]
            if v < floor:
                ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1, facecolor=HATCH_FACE,
                                       hatch=HATCH, lw=0, zorder=2))
                continue
            c = _text_color(v)
            s = subs[i][k] if subs else None
            ax.text(k, i - (0.15 if s else 0.0), f"{v:.2f}"[1:] if v < 1 else "1",
                    ha="center", va="center", fontsize=fs, color=c, zorder=3)
            if s:
                ax.text(k, i + 0.28, s, ha="center", va="center", fontsize=0.68 * fs,
                        color=c, alpha=0.85, zorder=3)
    for k in np.arange(0.5, m - 0.5):
        ax.axvline(k, color=HM_GRID_COLOR, lw=0.7, zorder=2.5)
    for i in np.arange(0.5, n - 0.5):
        ax.axhline(i, color=HM_GRID_COLOR, lw=0.7, zorder=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which="both", left=False, bottom=False, top=False, right=False)
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_linewidth(aa.LW["axes"]); sp.set_color(HM_FRAME_COLOR); sp.set_zorder(4)
    return ax


def _hm_colorbar(fig, W, H, x, y0, h, w, floor):
    cax = fig.add_axes([x / W, 1 - (y0 + h) / H, w / W, h / H])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=HM_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "", "0.5", "", "1"])
    cax.minorticks_off()
    cax.tick_params(which="both", direction="in", left=True, right=True,
                    length=0.6 * aa.TICK["major"], width=aa.LW["axes"], pad=1.2,
                    labelsize=FS_L["tick"] - 1)
    if floor > 0:
        cax.axhspan(0, floor, facecolor=HATCH_FACE, hatch=HATCH, lw=0, zorder=3)
    cb.outline.set_linewidth(aa.LW["axes"]); cb.outline.set_edgecolor(HM_FRAME_COLOR)
    cb.set_label(CB_LABEL, fontsize=FS_L["label"] - 1, labelpad=3.2, color=INK)
    return cb


def _floor_legend(fig, W, H, x_mid_mm):
    fig.legend([Patch(facecolor=HATCH_FACE, hatch=HATCH, edgecolor=INK, lw=0.5)],
               [FLOOR_TEXT], loc="lower center", bbox_to_anchor=(x_mid_mm / W, 0.3 / H),
               fontsize=FS_L["legend"] - 1.5, frameon=False, handlelength=1.4,
               handleheight=0.9, borderaxespad=0.0)


def _z_header(fig, W, H, xs_mm, y_mm, fs):
    for x, z in zip(xs_mm, Z_PLOT):
        fig.text(x / W, 1 - y_mm / H, f"{float(z):.1f}", ha="center", va="baseline",
                 fontsize=fs, color=INK)


def plot_mi_heat_z(rI, cols, name="summary_mi_column", vox=None, floor=0.0):
    """Full r_I table: rows = statistics (PDF l2..tau6, PS k1..k5, single voxel at 100/50/25 h if vox is ..."""
    npar, nz = rI.shape[0], len(Z_PLOT)
    blocks = [("PDF", PDF_NAMES), ("PS", PS_NAMES)]
    if vox:
        blocks.append(("vox", [f"{h:g} h" for h in VOX_HOURS]))
    blocks.append(("mean", [FAMILY["mean"]]))

    def matrix(b, p):
        if b == "vox":
            return np.array([[vox[z]["r"][s, p] for z in Z_PLOT]
                             for s in range(len(VOX_HOURS))])
        return _family_rI(rI, cols, p, b)

    W = COLUMN_MM
    x_fam, x_stat, hm_x = 2.4, 10.6, 11.6
    cb_gap, cb_w, right = 1.6, 1.8, 9.6
    pgap, rh, bgap = 1.4, 4.0, 1.6
    y_par, y_z, y_hm = 3.6, 7.6, 8.8
    foot = 5.6 if floor > 0 else 1.2
    hm_w = W - hm_x - cb_gap - cb_w - right
    cw = (hm_w - (npar - 1) * pgap) / (npar * nz)
    nrow = sum(len(n) for _, n in blocks)
    H = y_hm + nrow * rh + (len(blocks) - 1) * bgap + foot
    letters = iter("abcdefgh")

    with mpl.rc_context(aa.rc()), mpl.rc_context(HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        y = y_hm
        for b, names in blocks:
            n = len(names)
            for p in range(npar):
                _hm_block(fig, W, H, hm_x + p * (nz * cw + pgap), y, nz * cw, n * rh,
                          matrix(b, p), floor)
            for s, lab in enumerate(names):
                fig.text(x_stat / W, 1 - (y + (s + 0.5) * rh) / H, lab, ha="right",
                         va="center_baseline", color=INK,
                         fontsize=FS_L["tick"] - (1.5 if b == "vox" else 0))
            if b != "mean":
                fig.text(x_fam / W, 1 - (y + n * rh / 2) / H,
                         rf"$\mathbf{{({next(letters)})}}$ {FAMILY[b]}", rotation=90,
                         ha="center", va="center", fontsize=FS_L["family"] - 0.5,
                         weight="bold", color=INK)
            y += n * rh + bgap
        y_end = y - bgap

        for p in range(npar):
            x0 = hm_x + p * (nz * cw + pgap)
            fig.text((x0 + nz * cw / 2) / W, 1 - y_par / H, PARAMS[p], ha="center",
                     va="baseline", fontsize=FS_L["tick"], color=INK)
            _z_header(fig, W, H, [x0 + (k + 0.5) * cw for k in range(nz)], y_z,
                      FS_L["tick"] - 2.0)
        fig.text((hm_x - 0.6) / W, 1 - y_z / H, r"$z$", ha="right", va="baseline",
                 fontsize=FS_L["tick"] - 0.5, color=INK)

        _hm_colorbar(fig, W, H, hm_x + hm_w + cb_gap, y_hm, y_end - y_hm, cb_w, floor)
        if floor > 0:
            _floor_legend(fig, W, H, hm_x + hm_w / 2)
        save(fig, name)
        plt.close(fig)
    return W, H


def plot_mi_5x9(rI, cols, name="summary_mi_best_column", floor=0.0):
    """Compact r_I table for the main text: rows = parameters, three framed blocks (a) PDF | (b) PS | (c) ..."""
    npar, nz = rI.shape[0], len(Z_PLOT)
    fams = [("PDF", PDF_NAMES), ("PS", PS_NAMES), ("mean", None)]

    W = COLUMN_MM
    x_par, hm_x = 13.2, 14.0
    cb_gap, cb_w, right, fgap = 1.4, 1.8, 9.6, 2.0
    rh = 6.2
    y_fam, y_z, y_hm = 3.8, 7.4, 8.6
    foot = 5.4 if floor > 0 else 1.2
    hm_w = W - hm_x - cb_gap - cb_w - right
    cw = (hm_w - (len(fams) - 1) * fgap) / (len(fams) * nz)
    H = y_hm + npar * rh + foot

    with mpl.rc_context(aa.rc()), mpl.rc_context(HATCH_RC):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        for f, (b, names) in enumerate(fams):
            M = np.zeros((npar, nz))
            subs = [[None] * nz for _ in range(npar)]
            for j in range(npar):
                A = _family_rI(rI, cols, j, b)
                best = A.argmax(0)
                M[j] = A[best, np.arange(nz)]
                if names:
                    for k in range(nz):
                        if M[j, k] >= floor:
                            subs[j][k] = names[best[k]]
            x0 = hm_x + f * (nz * cw + fgap)
            _hm_block(fig, W, H, x0, y_hm, nz * cw, npar * rh, M, floor,
                      subs if names else None, fs=7.3)
            fig.text((x0 + nz * cw / 2) / W, 1 - y_fam / H,
                     rf"$\mathbf{{({'abc'[f]})}}$ {FAMILY[b]}", ha="center",
                     va="baseline", fontsize=FS_L["family"] - 0.5, weight="bold",
                     color=INK)
            _z_header(fig, W, H, [x0 + (k + 0.5) * cw for k in range(nz)], y_z,
                      FS_L["tick"] - 1.5)
        fig.text((hm_x - 0.6) / W, 1 - y_z / H, r"$z$", ha="right", va="baseline",
                 fontsize=FS_L["tick"] - 0.5, color=INK)
        for j in range(npar):
            fig.text(x_par / W, 1 - (y_hm + (j + 0.5) * rh) / H, PARAMS[j], ha="right",
                     va="center_baseline", fontsize=FS_L["label"] - 0.5, color=INK)

        _hm_colorbar(fig, W, H, hm_x + hm_w + cb_gap, y_hm, npar * rh, cb_w, floor)
        if floor > 0:
            _floor_legend(fig, W, H, hm_x + hm_w / 2)
        save(fig, name)
        plt.close(fig)
    return W, H


def plot_mi_large(rI, cols, name="summary_mi_dots_column"):
    """r_I dot plot: one row per parameter (label once, left), columns PDF | PS | <dTb> (title once, top) ..."""
    W = COLUMN_MM
    left, right, gap = 10.6, 1.0, 1.8
    top_leg, top_fam, row_h, row_gap, bottom = 0.5, 9.4, 15.0, 1.3, 6.4
    npar = rI.shape[0]
    y_ax = top_fam + 1.0
    H = y_ax + npar * row_h + (npar - 1) * row_gap + bottom
    unit = (W - left - right - 2 * gap) / (5 + 5 + 1.5)
    panels = [("PDF", 5), ("PS", 5), ("mean", 1.5)]

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        x0 = left
        for b, nslot in panels:
            pw = nslot * unit
            for j in range(npar):
                y0 = y_ax + j * (row_h + row_gap)
                ax = fig.add_axes([x0 / W, 1 - (y0 + row_h) / H, pw / W, row_h / H])
                last = j == npar - 1
                if b == "mean":
                    idx = [i for i, c in enumerate(cols) if c[2] == "mean"]
                    for k, z in enumerate(Z_PLOT):
                        st = _zstyle(z); st["lw"] = 0
                        ax.plot([2.2 * DODGE[z]], [rI[j, idx[k]]], zorder=3, **st)
                    ax.set_xlim(-0.75, 0.75)
                    ax.set_xticks([])
                    ax.tick_params(axis="x", which="both", bottom=False, top=False)
                else:
                    for z in Z_PLOT:
                        idx = [i for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                        ax.plot(np.arange(5) + DODGE[z], rI[j, idx], zorder=3, **_zstyle(z))
                    _xstats(ax, PDF_NAMES if b == "PDF" else PS_NAMES, last)
                ax.set_ylim(-0.06, 1.06)
                ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
                ax.yaxis.set_minor_locator(FixedLocator([0.25, 0.75]))
                ax.tick_params(axis="y", labelsize=FS_L["tick"], pad=1.5)
                if b == "PDF":
                    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
                    ax.set_ylabel(PARAMS[j], fontsize=FS_L["label"], labelpad=2.0)
                else:
                    ax.tick_params(axis="y", which="both", labelleft=False)
            _family_title(fig, W, H, x0 + pw / 2, top_fam, b)
            x0 += pw + gap
        _z_legend(fig, W, H, left, W - right, top_leg)
        fig.text(1.0 / W, 1 - (top_fam) / H, r"$r_I$", ha="left", va="baseline",
                 fontsize=FS_L["label"] + 1, color=INK)
        save(fig, name)
        plt.close(fig)
    return W, H


def plot_ic_dots(Ic, cols, name="summary_ic_dots_small_column"):
    """Two panels (PDF | PS) sharing a log I_c axis."""
    annot, tick, label = FS["annot"], FS["tick"], FS["label"]
    zcol = DOTS_Z_COLORS
    zmark = dict(zip(Z_PLOT, ["o", "s", "^"]))
    dodge = dict(zip(Z_PLOT, [-0.16, 0.0, 0.16]))

    W = COLUMN_MM
    left, right, gap, top, bottom = 9.0, 0.6, 1.4, 4.6, 4.4
    ax_h = 24.0
    H = top + ax_h + bottom
    pw = (W - left - right - gap) / 2
    keep = [i for i, c in enumerate(cols) if c[2] != "mean"]
    lo, hi = 0.4, 10 ** (np.log10(np.max(np.asarray(Ic)[keep])) + 0.25)

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        handles = {}
        for p, b in enumerate(("PDF", "PS")):
            ax = fig.add_axes([(left + p * (pw + gap)) / W, bottom / H, pw / W, ax_h / H])
            names = PDF_NAMES if b == "PDF" else PS_NAMES
            for z in Z_PLOT:
                idx = [i for i, c in enumerate(cols) if c[2] == b and c[3] == z]
                xs = np.arange(5) + dodge[z]
                h, = ax.plot(xs, np.asarray(Ic)[idx], color=zcol[z], lw=LW_THIN,
                             marker=zmark[z], ms=3.4, mec=WHITE, mew=0.4, zorder=3)
                handles[z] = h
            ax.axhline(1.0, color=IC_REF_COLOR, lw=LW_REF, ls=":", zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            ax.set_xlim(-0.5, 4.5)
            ax.set_xticks(range(5), names, fontsize=tick)
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.tick_params(axis="x", which="major", length=0.8 * aa.TICK["major"], pad=1.5)
            ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(0, int(np.log10(hi)) + 1)]))
            ax.tick_params(axis="y", labelsize=annot, pad=1.2)
            if p == 0:
                ax.set_ylabel(r"$I_c$", fontsize=label, labelpad=1.0)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False)
            ax.text(0.04, 0.13, FAMILY[b], transform=ax.transAxes, ha="left", va="bottom",
                    **family_text(b))
        fig.legend([handles[z] for z in Z_PLOT], [zlab(z) for z in Z_PLOT],
                   loc="upper center", bbox_to_anchor=((left + W - right) / 2 / W, 1.0),
                   ncol=3, fontsize=annot, handlelength=1.8, columnspacing=1.6,
                   handletextpad=0.4, borderaxespad=0.15)
        save(fig, name)
        plt.close(fig)
    return W, H


def plot_ic(Ic, cols, name="summary_ic_bars_column"):
    tiny, annot, tick, label = FS["tiny"], FS["annot"], FS["tick"], FS["label"]
    G_Z, G_FAM, MEAN_W, BAR = 0.45, 1.1, 1.9, 0.68

    x, w, pos, prev = [], [], 0.0, None
    for c in cols:
        key = (c[2], c[3] if c[2] != "mean" else "")
        if prev is not None and key != prev:
            pos += G_FAM if c[2] != prev[0] else G_Z
        sw = MEAN_W if c[2] == "mean" else 1.0
        x.append(pos + sw / 2)
        w.append(sw)
        pos += sw
        prev = key
    x, w = np.array(x), np.array(w)

    W = COLUMN_MM
    left, right, top, ax_h, below = 7.6, 0.3, 0.8, 17.0, 6.0
    H = top + ax_h + below
    lo = min(0.3, 0.5 * float(np.min(Ic)))

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        ax = fig.add_axes([left / W, below / H, (W - left - right) / W, ax_h / H])
        ax.bar(x, np.minimum(Ic, CLIP) - lo, bottom=lo, width=BAR * w ** 0.35,
               color=[COL[c[2]] for c in cols], lw=0, zorder=2)
        for xi, v in zip(x, Ic):
            if v > CLIP:
                e = int(np.floor(np.log10(v)))
                ax.text(xi, CLIP * 0.62, rf"${v / 10**e:.0f}\!\times\!10^{{{e}}}$",
                        rotation=90, ha="center", va="top", fontsize=tiny, color=WHITE,
                        zorder=4)
        ax.axhline(1.0, color=IC_REF_COLOR, lw=LW_REF, ls=":", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(lo, CLIP)
        ax.set_xlim(x[0] - 0.75, x[-1] + w[-1] / 2 + 0.2)

        ax.yaxis.set_major_locator(FixedLocator([1, 1e2, 1e4]))
        ax.yaxis.set_minor_locator(FixedLocator([1e1, 1e3]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="both", direction="out", right=False, pad=1.2,
                       labelsize=annot)
        ax.tick_params(axis="y", which="major", length=0.8 * aa.TICK["major"])
        ax.tick_params(axis="y", which="minor", length=0.8 * aa.TICK["minor"])
        ax.set_ylabel(r"$I_c$", fontsize=label, labelpad=0.8)
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        tr = ax.get_xaxis_transform()
        dy = 1.0 / ax_h
        for xi, c in zip(x, cols):
            txt = c[1] if c[2] != "mean" else zlab(f"{float(c[3]):.1f}", eq=False)
            ax.text(xi, -0.35 * dy, txt, transform=tr, ha="center", va="top",
                    fontsize=tiny, color=INK)
        for b, z, a, e in groups(cols):
            xc = (x[a] + x[e]) / 2
            txt = zlab(z) if b != "mean" else r"$\langle\delta T_b\rangle$"
            ax.text(xc, -3.0 * dy, txt, transform=tr, ha="center", va="top",
                    fontsize=tiny, color=INK)
        for b in ("PDF", "PS"):
            xs = [xi for xi, c in zip(x, cols) if c[2] == b]
            ax.text((min(xs) + max(xs)) / 2, 0.985, FAMILY[b], transform=tr,
                    ha="center", va="top", **family_text(b))

        save(fig, name)
        plt.close(fig)
    return W, H


def plot_mi(rI, cols, name="summary_mi_heatmap_column"):
    """Cells just larger than the 6.5 pt values in them (~7.3 x 2.75 mm)."""
    annot, tick = FS["annot"], FS["tick"]
    M = rI.T
    npar = M.shape[1]
    G = groups(cols)

    W = COLUMN_MM
    rh, g_z, g_fam = 2.75, 0.5, 3.4
    head, bottom, right, gap_p = 13.2, 0.5, 0.3, 2.4
    x_z, x_stat, gut = 1.6, 6.1, 6.8
    P = (W - gap_p - right) / 2
    cw = (P - gut) / npar
    panel_x = {"PDF": 0.0, "PS": P + gap_p, "mean": P + gap_p}

    spans, y_of = [], {"PDF": head, "PS": head}
    for b, z, a, e in G:
        side = "PDF" if b == "PDF" else "PS"
        y = y_of[side]
        if b == "mean":
            y += g_fam
        elif y > head:
            y += g_z
        spans.append((b, z, a, e, y, y + (e - a + 1) * rh))
        y_of[side] = spans[-1][5]
    y_cb = y_of["PDF"] + g_fam
    H = max(y_of["PS"], y_cb + 1.6 + 6.0) + bottom

    def fx(x):
        return x / W

    def fy(y):
        return 1 - y / H

    with mpl.rc_context(aa.rc()):
        fig = plt.figure(figsize=(W * aa.MM, H * aa.MM))
        im = None
        for b, z, a, e, y0, y1 in spans:
            x0 = panel_x[b] + gut
            ax = fig.add_axes([fx(x0), fy(y1), cw * npar / W, (y1 - y0) / H])
            block = M[a:e + 1]
            im = ax.imshow(block, aspect="auto", cmap=OLD_HEATMAP_CMAP, vmin=0, vmax=1,
                           extent=(-0.5, npar - 0.5, len(block) - 0.5, -0.5))
            ax.set_autoscale_on(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(which="both", left=False, bottom=False, top=False, right=False)
            ax.minorticks_off()
            for sp in ax.spines.values():
                sp.set_visible(False)
            for i in range(len(block)):
                for j in range(npar):
                    v = block[i, j]
                    ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center",
                            va="center_baseline", fontsize=annot,
                            color=WHITE if v > OLD_HEATMAP_TEXT_SWITCH else INK)
            for i, r in enumerate(range(a, e + 1)):
                yc = fy(y0 + (i + 0.5) * rh)
                txt = zlab(cols[r][3], eq=False) if b == "mean" else cols[r][1]
                fig.text(fx(panel_x[b] + x_stat), yc, txt, ha="right",
                         va="center_baseline", fontsize=annot, color=INK)
            if b != "mean":
                fig.text(fx(panel_x[b] + x_z), fy((y0 + y1) / 2), zlab(z), rotation=90,
                         ha="center", va="center", fontsize=annot, color=INK)
            else:
                fig.text(fx(panel_x[b] + 0.3), fy(y0 - 0.55), FAMILY[b], ha="left",
                         va="baseline", **family_text(b))

        for side in ("PDF", "PS"):
            x0 = panel_x[side] + gut
            for j, p in enumerate(PARAMS[:npar]):
                fig.text(fx(x0 + (j + 0.5) * cw), fy(head - 0.7), p, rotation=90,
                         ha="center", va="bottom", fontsize=tick, color=INK)
            fig.text(fx(panel_x[side] + 0.3), fy(head - 0.7), FAMILY[side], ha="left",
                     va="bottom", **family_text(side))

        x0 = panel_x["PDF"] + gut
        cax = fig.add_axes([fx(x0 + 0.5 * cw), fy(y_cb + 1.6), (npar - 1) * cw / W, 1.6 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "", "0.5", "", "1"])
        cb.ax.tick_params(which="both", direction="out", length=0.6 * aa.TICK["major"],
                          width=aa.LW["axes"], pad=0.8, labelsize=annot)
        cb.ax.minorticks_off()
        cb.outline.set_linewidth(aa.LW["axes"])
        fig.text(fx(x0 - 0.6), fy(y_cb + 0.8), r"$r_I$", ha="right", va="center",
                 fontsize=tick, color=INK)

        save(fig, name)
        plt.close(fig)
    return W, H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--only", choices=["ic", "mi"], default=None)
    ap.add_argument("--ic-style", choices=["large", "dots", "bars"], default="large",
                    help="large (default): large dot plot; dots: small; bars: clipped bars")
    ap.add_argument("--vox", type=int, default=0,
                    help="simulations for the voxel I_c box plots (default 0 = panel off)")
    ap.add_argument("--vox-seed", type=int, default=0)
    ap.add_argument("--vox-cache", type=Path, default=VOX_CACHE)
    ap.add_argument("--recompute-vox", action="store_true")
    ap.add_argument("--vox-mi", type=int, default=20,
                    help="voxel positions of the single-voxel r_I run, used for the "
                         "estimator floor (0 = off; then give --floor)")
    ap.add_argument("--vox-mi-cache", type=Path, default=VOX_MI_CACHE)
    ap.add_argument("--vox-block", action="store_true",
                    help="add the single-voxel rows to the full r_I table (default: off, "
                         "the voxel run only sets the estimator floor)")
    ap.add_argument("--recompute-vox-mi", action="store_true")
    ap.add_argument("--mi-style", choices=["both", "heatz", "best", "dots", "heatmap"],
                    default="both",
                    help="both (default): heatz + best; heatz: full table, statistics as "
                         "rows (+ single-voxel block); best: compact 5 x 9 table, best "
                         "statistic per family; dots: r_I dot plot; heatmap: old table")
    ap.add_argument("--floor", type=float, default=None,
                    help="r_I estimator floor; cells below it are hatched. Default: the "
                         "shuffled-parameter floor of the single-voxel run, else 0")
    a = ap.parse_args()
    aa.use()
    rI, Ic, n_used, cols = load(a.cache)
    print(f"MI from {n_used} simulations, one noise realisation each")

    if a.only in (None, "ic"):
        if a.ic_style == "large":
            vox = (compute_vox_ic(a.vox, a.vox_seed, a.vox_cache, a.recompute_vox)
                   if a.vox > 0 else None)
            print("  ic: %.0f x %.0f mm" % plot_ic_large(Ic, cols, vox=vox))
        else:
            f = {"dots": plot_ic_dots, "bars": plot_ic}[a.ic_style]
            print("  ic: %.0f x %.0f mm" % f(Ic, cols))
        m = [v for v, c in zip(Ic, cols) if c[2] == "mean"]
        print("  <dTb> I_c (z = " + ", ".join(Z_PLOT) + "): " +
              ", ".join(f"{v:.1e}" for v in m) + "  -> caption")

    if a.only in (None, "mi"):
        if a.mi_style in ("both", "heatz", "best"):
            need_vox = a.vox_mi > 0 and (a.mi_style != "best" or a.floor is None)
            vmi = (compute_vox_mi(a.vox_mi, cache=a.vox_mi_cache, recompute=a.recompute_vox_mi)
                   if need_vox else None)
            if a.floor is not None:
                floor = a.floor
            elif vmi:
                floor = float(np.mean([vmi[z]["null"] for z in Z_PLOT]))
            else:
                floor = 0.0
            print(f"  r_I estimator floor (hatched below): {floor:.2f}  -> caption")
            if a.mi_style in ("both", "heatz"):
                print("  mi (full): %.0f x %.0f mm" % plot_mi_heat_z(
                    rI, cols, vox=vmi if a.vox_block else None, floor=floor))
            if a.mi_style in ("both", "best"):
                print("  mi (best): %.0f x %.0f mm" % plot_mi_5x9(rI, cols, floor=floor))
        else:
            f = {"dots": plot_mi_large, "heatmap": plot_mi}[a.mi_style]
            print("  mi: %.0f x %.0f mm" % f(rI, cols))


if __name__ == "__main__":
    main()
