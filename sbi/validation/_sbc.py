import os
import math
import re
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import _settings as _S
from _settings import (VIVID, FIGFMT, get_rc, get_preset, resolve_styles, tex, with_suffix, save_kwargs,
                       group_color_map, sbc_paper_presets)

BAND_COLOR = "#4C72B0"
BAND_ALPHA = (0.22, 0.11)
REF_COLOR  = "#222222"
UNIT_COLOR = "#8C8C8C"

def _sbc_reference_curves(nbins, n_used):
    s = math.sqrt(nbins * (1.0 - 1.0 / nbins) / n_used) if n_used > 0 else 0.0
    bands = {"low1": max(1 - s, 0), "high1": 1 + s, "low2": max(1 - 1.96 * s, 0), "high2": 1 + 1.96 * s}
    std = NormalDist()
    u = np.linspace(0.0, 1.0, 401)
    width = 1.0 / nbins

    def _expected(loc, scale):
        F = NormalDist(loc, scale).cdf
        Fz = lambda zz: 0.0 if zz == -math.inf else (1.0 if zz == math.inf else F(zz))
        lo = np.clip(u - width / 2, 0, 1 - width)
        out = []
        for a in lo:
            za, zb = std.inv_cdf(a) if a > 0 else -math.inf, std.inv_cdf(a + width) if a + width < 1 else math.inf
            out.append((Fz(zb) - Fz(za)) / width)
        return u, np.array(out)
    return bands, _expected(0.2, 1.0), _expected(0.0, 0.85)

def _grid(n, P):
    if P["name"] == "beamer":
        ncols = 1 if n <= 3 else 2
        nrows = math.ceil(n / ncols)
        return nrows, ncols, P["figsize"]
    if P.get("sbc_layout") == "row":
        nrows, ncols = 1, n
        h = (P["fig_width"] - 0.38) / n * 0.8 + 1.05
        return nrows, ncols, (P["fig_width"], h)
    ncols = 1 if n <= 4 else 2
    nrows = math.ceil(n / ncols)
    h = P["sbc_panel_height"] * nrows + 1.05
    return nrows, ncols, (P["fig_width"], h)

def _render_one(mode, arms, out_path, nbins, common_params, labels, P):
    AA = bool(P.get("aa"))
    if AA:
        labels = P["labels"]
    n = len(common_params)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    n_used = max((r["used"] for r in arms), default=0)
    bands, bias_hist, under_hist = _sbc_reference_curves(nbins, n_used)

    ymax = max(2.0, bands["high2"] * 1.05, max(np.max(np.asarray(r["rank_hist"])[:n]) for r in arms) * 1.08 if arms else 0)
    ymax = math.ceil(ymax * 2) / 2

    unique_groups = list(dict.fromkeys(a.get("group", "default") for a in arms))
    group_colors = group_color_map(unique_groups)
    gst = {g: _S.aa_group_style(g, group_colors[g]) for g in unique_groups} if AA else {}
    if AA:
        pooled_all = [np.mean([np.asarray(r["rank_hist"])[:n] for r in arms
                               if r.get("group", "default") == g], axis=0) for g in unique_groups]
        vals = np.concatenate([p.ravel() for p in pooled_all])
        if _S.SHOW_SEEDS:
            vals = np.concatenate([vals] + [np.asarray(r["rank_hist"])[:n].ravel() for r in arms])
        ylo = min(bands["low2"], vals.min()); yhi = max(bands["high2"], vals.max())
        pad = 0.06 * (yhi - ylo)
        ylo = math.floor((ylo - pad) * 20) / 20; yhi = math.ceil((yhi + pad) * 20) / 20

    nrows, ncols, figsize = _grid(n, P)
    with plt.rc_context(get_rc(P)):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        W, H = figsize
        if P["name"] == "beamer":
            fig.subplots_adjust(left=0.075, right=0.735, top=0.965, bottom=0.135, hspace=0.07, wspace=0.05)
        elif AA:
            fig.subplots_adjust(left=0.38 / W, right=0.985, top=1 - 0.04 / H,
                                bottom=0.38 / H, hspace=0.07, wspace=0.05)
        else:
            n_leg = len(dict.fromkeys(a.get("group", "default") for a in arms)) + 4
            leg_rows = math.ceil(n_leg / P.get("legend_ncol", 4))
            top_in = 0.10 + 0.135 * leg_rows
            fig.subplots_adjust(left=0.38 / W, right=0.985, top=1 - top_in / H,
                                bottom=0.38 / H, hspace=0.07, wspace=0.05)

        flat = axes.ravel(order="F") if ncols > 1 else axes.ravel()
        for ax in flat[n:]: ax.set_visible(False)

        for jj, j in enumerate(common_params):
            ax = flat[jj]
            ax.fill_between([0, 1], bands["low2"], bands["high2"], color=BAND_COLOR, alpha=BAND_ALPHA[1], lw=0, zorder=0)
            ax.fill_between([0, 1], bands["low1"], bands["high1"], color=BAND_COLOR, alpha=BAND_ALPHA[0], lw=0, zorder=0)
            ax.axhline(1.0, color=UNIT_COLOR, lw=P["lw_axes"] * 0.8, ls=P.get("dash", (0, (5, 4))), zorder=1)

            for grp in unique_groups:
                grp_arms = [r for r in arms if r.get("group", "default") == grp]
                col = group_colors[grp]

                all_hists = []
                for r in grp_arms:
                    rh = np.asarray(r["rank_hist"])[jj]
                    all_hists.append(rh)
                    if len(grp_arms) > 1 and _S.SHOW_SEEDS:
                        ax.stairs(rh, edges, lw=P["lw_seed"] if AA else P["lw_hist"] * 0.6, color=col, ls="--", alpha=0.5,
                                  joinstyle="miter", capstyle="butt", baseline=None, zorder=4)

                if all_hists and AA:
                    pooled_rh = np.mean(all_hists, axis=0)
                    g = gst[grp]
                    if g["fill"]:
                        ax.stairs(pooled_rh, edges, baseline=1.0, fill=True, color=_S.AA_REF_FILL,
                                  lw=0, zorder=2)
                        ax.stairs(pooled_rh, edges, color=_S.AA_REF_EDGE, lw=0.4,
                                  baseline=None, joinstyle="miter", capstyle="butt", zorder=2.5)
                    else:
                        ax.stairs(pooled_rh, edges, lw=_S.AA_LW_SBC, color=g["color"], ls=g["ls"],
                                  joinstyle="miter", capstyle="butt", baseline=None, zorder=5)
                elif all_hists:
                    pooled_rh = np.mean(all_hists, axis=0)
                    ax.stairs(pooled_rh, edges, lw=P["lw_hist"] if AA else P["lw_hist"] * 1.6, color=col, ls="-", alpha=1.0,
                              joinstyle="miter", capstyle="butt", baseline=None, zorder=5)

            ax.plot(*bias_hist, ls=P.get("dash", "--"), lw=P["lw_ref"], color=REF_COLOR, zorder=3)
            ax.plot(*under_hist, ls=":", lw=P["lw_ref"], color=REF_COLOR, zorder=3)

            ax.set_xlim(0, 1); ax.tick_params(labelbottom=False)
            ax.set_ylim(*((ylo, yhi) if AA else (0, ymax)))
            if AA:
                _S._aa.panel_label(ax, "abcdefgh"[jj], labels[j])
            else:
              ax.text(0.015, 0.93, tex(labels[j], P), transform=ax.transAxes, fontsize=P["fs_label"], va="top", ha="left", zorder=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

            row, col = (jj % nrows, jj // nrows) if ncols > 1 else (jj, 0)
            if AA:
                yt, ymin_step = _S.nice_ticks(ylo, yhi, n=3, margin=0.05)
                ax.set_yticks(yt); ax.yaxis.set_minor_locator(MultipleLocator(ymin_step))
            else:
                ystep = 0.5 if ymax <= 2.5 else 1.0
                ax.set_yticks(np.arange(0, ymax - 1e-9, ystep))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            has_right = col < ncols - 1 and (col + 1) * nrows + row < n
            ax.set_xticks(np.arange(0, 0.9 if has_right else 1.0 + 1e-9, 0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))

            if col > 0: ax.tick_params(labelleft=False)
            if row == nrows - 1 or jj + 1 >= n:
                ax.tick_params(labelbottom=True); ax.set_xlabel("rank CDF value", labelpad=P["pad_label"])

        fig.supylabel("normalised rank density", fontsize=P["fs_label"], x=0.005)

        handles, leg_labels = [], []
        for grp in unique_groups:
            handles.append(Line2D([], [], color=group_colors[grp], ls="-", lw=P["lw_hist"] if AA else P["lw_hist"] * 1.5, solid_capstyle="butt"))
            leg_labels.append(tex(_S._ps.arm_label(grp) if _S._ps is not None else grp, P))

        handles += [Line2D([], [], color=REF_COLOR, ls=P.get("dash", "--"), lw=P["lw_ref"]), Line2D([], [], color=REF_COLOR, ls=":", lw=P["lw_ref"]), Patch(facecolor=BAND_COLOR, alpha=BAND_ALPHA[0], lw=0), Patch(facecolor=BAND_COLOR, alpha=BAND_ALPHA[1], lw=0)]
        leg_labels += [r"bias (0.2$\sigma$)", ("under-confidence\n" r"(0.15$\sigma$)" if P["name"] == "beamer" else r"under-confidence (0.15$\sigma$)"), tex("68% interval", P), tex("95% interval", P)]

        if AA:
            from matplotlib.patches import Patch as _Patch
            ents = []
            for grp in unique_groups:
                g = gst[grp]
                h = (_Patch(facecolor=_S.AA_REF_FILL, edgecolor=_S.AA_REF_EDGE, lw=_S._aa.LW["contour"])
                     if g["fill"] else Line2D([], [], color=g["color"], ls=g["ls"], lw=_S.AA_LW_SBC))
                ents.append((g["header"] or g["short"], g["short"] if g["header"] else "", h))
            ref_cols = [[(r"bias ($0.2\sigma$)", handles[-4]),
                         (r"under-confidence ($0.15\sigma$)", handles[-3])],
                        [("68% interval", handles[-2]), ("95% interval", handles[-1])]]
            H_, L_, nc, hi = _S.grouped_legend(ents, ref_cols, P)
            _S.save_grouped_legend(H_, L_, nc, hi, _S.legend_path(out_path), P)
        elif P["name"] == "beamer":
            fig.legend(handles, leg_labels, loc="upper left", bbox_to_anchor=(0.755, 0.965), borderaxespad=0.0, handlelength=2.4, handletextpad=0.7, labelspacing=0.7, frameon=False)
            if mode: fig.text(0.755, 0.135, tex(mode, P), ha="left", va="bottom", fontsize=P["fs_tick"], color="0.35")
        else:
            fig.legend(handles, leg_labels, loc="lower center", bbox_to_anchor=(0.53, 1 - (top_in - 0.04) / H), ncol=min(len(handles), P.get("legend_ncol", 4)), borderaxespad=0.0, handlelength=2.2, handletextpad=0.5, columnspacing=1.2, labelspacing=0.35, frameon=False)

        fig.savefig(out_path, dpi=300, **save_kwargs(P))
        plt.close(fig)
    print(f"[sbc] wrote {out_path}")
    return out_path

def plot_sbc(report, out, nbins, common_params, labels, style=None):
    styles = resolve_styles(style)
    paths = []
    for mode, arms in report.items():
        if not arms: continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", mode)
        base = os.path.join(out, f"sbc_overlay_{safe}.{FIGFMT}")
        for st in styles:
            variants = sbc_paper_presets() if st == "paper" else []
            if variants:
                root, ext = os.path.splitext(base)
                for tag, P in variants:
                    paths.append(_render_one(mode, arms, f"{root}_{tag}{ext}", nbins, common_params, labels, P))
                continue
            P = get_preset(st)
            paths.append(_render_one(mode, arms, with_suffix(base, st, len(styles)), nbins, common_params, labels, P))
    return paths
