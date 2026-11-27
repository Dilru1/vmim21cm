import os
import sys
import re
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import _settings as _S
from _settings import (VIVID, TRUTH_COLOR, CONTOUR_LEVEL, CONTOUR_LABEL, FIGFMT,
                       get_rc, get_preset, resolve_styles, tex,
                       with_suffix, save_kwargs, group_color_map)

def credible_levels(H, fracs=(0.68, 0.95), total=None):
    """Density thresholds enclosing the given probability fractions (ascending)."""
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= (total if total else csum[-1])
    out = []
    for f in fracs:
        i = np.searchsorted(csum, f)
        out.append(flat[min(i, len(flat) - 1)])
    return sorted(out)

def _smooth2d(H, sigma):
    if np.all(np.asarray(sigma) <= 0):
        return H
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H, sigma)
    except ImportError:
        return H

def _hist1d(ax, x, rng, bins, col, ls, lw, zorder, alpha=1.0, horizontal=False, fill=False):
    if fill:
        ax.hist(x, bins=bins, range=rng, density=True, histtype="stepfilled",
                orientation="horizontal" if horizontal else "vertical",
                fc=_S.AA_REF_FILL, ec=_S.AA_REF_EDGE, lw=0.4, zorder=2)
        return
    ax.hist(x, bins=bins, range=rng, density=True, histtype="step",
            orientation="horizontal" if horizontal else "vertical",
            lw=lw, color=col, ls=ls, alpha=alpha, zorder=zorder)

def _contour2d(ax, x, y, xr, yr, bins, smooth, col, ls, lw, zorder, alpha=1.0, fill=False):
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    H = _smooth2d(H.astype(float), smooth)
    if H.sum() <= 0:
        return
    lv = credible_levels(H, (CONTOUR_LEVEL,), total=len(x))[0]
    if not (0 < lv < H.max()):
        return
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if fill:
        ax.contourf(xc, yc, H.T, levels=[lv, H.max() * 1.01], colors=[_S.AA_REF_FILL],
                    antialiased=True, zorder=2)
        ax.contour(xc, yc, H.T, levels=[lv], colors=[_S.AA_REF_EDGE],
                   linewidths=0.4, antialiased=True, zorder=2.5)
        return
    ax.contour(xc, yc, H.T, levels=[lv], colors=[col], linewidths=lw,
               linestyles=[ls], alpha=alpha, antialiased=True, zorder=zorder)

def _despine_1d(ax, horizontal=False):
    ax.set_yticks([]) if not horizontal else ax.set_xticks([])
    for sp in (["top", "right", "left"] if not horizontal else ["top", "right", "bottom"]):
        ax.spines[sp].set_visible(False)
    if horizontal:
        ax.tick_params(axis="x", which="both", top=False, bottom=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", right=False)
    else:
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.tick_params(axis="x", which="both", top=False)

def _inner_ticks(axis, lim, nt, margin=0.12):
    lo, hi = lim
    if _AA_TICKS[0]:
        from matplotlib.ticker import MultipleLocator
        t, mstep = _S.nice_ticks(lo, hi, n=3, margin=max(margin, 0.08), nmax=3)
        axis.set_ticks(t)
        axis.set_minor_locator(MultipleLocator(mstep))
        return
    loc = MaxNLocator(nt, min_n_ticks=2)
    span = hi - lo
    t = [v for v in loc.tick_values(lo, hi) if lo + margin * span <= v <= hi - margin * span]
    if len(t) < 2:
        t = [v for v in MaxNLocator(2).tick_values(lo, hi) if lo + 0.05 * span <= v <= hi - 0.05 * span]
    axis.set_ticks(t)
    axis.set_minor_locator(AutoMinorLocator(2))

_AA_TICKS = [False]
FIXED_LIMITS = os.environ.get("PLOT_CORNER_FIXED_LIMITS", "1") != "0"
ZOOM_SET = os.environ.get("PLOT_CORNER_ZOOM", "1") != "0"


def _rotate(ax, P, axis="x"):
    rot = P["rotate_ticks"]
    if not rot:
        return
    if axis == "x":
        for t in ax.get_xticklabels():
            t.set_rotation(rot); t.set_ha("right"); t.set_rotation_mode("anchor")
    else:
        for t in ax.get_yticklabels():
            t.set_rotation(rot); t.set_va("top"); t.set_rotation_mode("anchor")

def _layout(fig, P, d):
    sp = P["panel_space"]
    W, H = fig.get_size_inches()
    if P["name"] == "beamer":
        bot, top, left = 0.15, 0.93, 0.085
        side = (top - bot) * H / W
        gs_main = fig.add_gridspec(d, d, left=left, right=left + side, bottom=bot, top=top,
                                   wspace=sp, hspace=sp)
        gs_zoom = None
        if d >= 3:
            zl = left + side + 0.11
            zside = min(0.975 - zl, side)
            gs_zoom = fig.add_gridspec(2, 2, left=zl, right=zl + zside, bottom=bot,
                                       top=bot + zside * W / H,
                                       width_ratios=[3, 1], height_ratios=[1, 3],
                                       wspace=sp, hspace=sp)
        return gs_main, gs_zoom, (left + side, top), "upper right", side * W / d

    l, r, b, t = 0.09, 0.985, 0.075, 0.985
    gs_main = fig.add_gridspec(d, d, left=l, right=r, bottom=b, top=t, wspace=sp, hspace=sp)
    gs_zoom = None
    if d >= 4:
        k = math.ceil(d / 2) / d
        mw, mh = r - l, t - b
        gs_zoom = fig.add_gridspec(2, 2, left=l + mw * k + 0.085, right=r,
                                   bottom=b + mh * k + 0.075, top=t,
                                   width_ratios=[3, 1], height_ratios=[1, 3],
                                   wspace=sp, hspace=sp)
    anchor = (l + (r - l) / d + 0.015, t)
    loc = "upper left" if d >= 3 else "upper right"
    if d < 3:
        anchor = (r, t)
    return gs_main, gs_zoom, anchor, loc, (r - l) * W / d

def _limits(arm_samples, common_params):
    lims = []
    for k in range(len(common_params)):
        lo = min(np.percentile(s[:, common_params[k]], 0.3) for _, _, s in arm_samples)
        hi = max(np.percentile(s[:, common_params[k]], 99.7) for _, _, s in arm_samples)
        if hi <= lo:
            hi, lo = lo + 0.5, lo - 0.5
        pad = 0.08 * (hi - lo)
        lims.append((lo - pad, hi + pad))
    return lims


def _drawn_limits(arm_samples, common_params, q=2.0, pad=0.15):
    groups = {}
    for grp, name, s in arm_samples:
        if "(Pooled)" in name:
            groups[grp] = s
        else:
            groups.setdefault(grp, s)
    out = []
    for k in range(len(common_params)):
        j = common_params[k]
        lo = min(np.percentile(s[:, j], q) for s in groups.values())
        hi = max(np.percentile(s[:, j], 100 - q) for s in groups.values())
        w = max(hi - lo, 1e-6)
        out.append((lo - pad * w, hi + pad * w))
    return out


def _render_one(arm_samples, truth, common_params, labels, out_path, P,
                bins, smooth, title, also_png, zoom_pair, fixed_lims=None):
    cols = VIVID
    AA = bool(P.get("aa"))
    if AA:
        labels = P["labels"]
    TDASH = P.get("dash", "--")
    _AA_TICKS[0] = AA
    d = len(common_params)
    L = lambda j: tex(labels[j], P)

    hide_identifiers = []

    group_n = {}
    for grp, name, _ in arm_samples:
        if "(Pooled)" not in name:
            group_n[grp] = group_n.get(grp, 0) + 1
    clean_samples = []
    for grp, name, s in arm_samples:
        if (any(hide in name for hide in hide_identifiers) or not _S.SHOW_SEEDS) \
                and "(Pooled)" not in name and group_n.get(grp, 0) > 1:
            if _S.SHOW_SEEDS:
                print(f"[corner] Hiding individual dashed contour for {name}")
            continue
        clean_samples.append((grp, name, s))

    arm_samples = clean_samples

    unique_groups = list(dict.fromkeys(grp for grp, _, _ in arm_samples))
    group_colors = group_color_map(unique_groups)

    group_counts = {g: sum(1 for grp, name, _ in arm_samples if grp == g and "(Pooled)" not in name) for g in unique_groups}

    styles = {}
    gst = {g: _S.aa_group_style(g, group_colors[g]) for g in unique_groups} if AA else {}
    fills = set()
    for grp, name, _ in arm_samples:
        col = group_colors[grp]

        if "(Pooled)" in name or group_counts[grp] == 1:
            f = 1.0 if AA else 1.6
            ls_ = gst[grp]["ls"] if AA else "-"
            if AA and gst[grp]["fill"]:
                fills.add(name)
            styles[name] = (col, ls_, P["lw_contour"] * f, P["lw_hist"] * f, 50 + unique_groups.index(grp), 1.0)
        else:
            styles[name] = (col, "--", P["lw_seed"] if AA else P["lw_contour"] * 0.5,
                            P["lw_seed"] if AA else P["lw_hist"] * 0.5, 10 + unique_groups.index(grp), 0.4)


    if P["name"] == "beamer":
        figsize = P["figsize"]
    else:
        w = min(P["fig_width"], 1.3 * d + 1.0)
        figsize = (w, w)

    with plt.rc_context(get_rc(P)):
        fig = plt.figure(figsize=figsize)
        gs_main, gs_zoom, leg_anchor, leg_loc, cell_in = _layout(fig, P, d)
        nt = P["corner_ticks"] if cell_in >= 0.9 else 2
        tmarg = float(np.clip(0.16 * P["fs_tick"] / 10.0 / cell_in, 0.08, 0.3))
        axes_main = [[None] * d for _ in range(d)]

        own = _limits(arm_samples, common_params)
        lims = list(fixed_lims) if fixed_lims is not None else own
        nb = [int(min(200, max(bins, round(bins * (lims[k][1] - lims[k][0])
                                           / max(own[k][1] - own[k][0], 1e-12)))))
              for k in range(d)]

        for r in range(d):
            for c in range(r + 1):
                ax = fig.add_subplot(gs_main[r, c])
                axes_main[r][c] = ax
                jr, jc = common_params[r], common_params[c]

                if r == c:
                    for grp, name, s in arm_samples:
                        col, ls, lw_c, lw_h, zorder, alpha = styles[name]
                        _hist1d(ax, s[:, jr], lims[r], nb[r], col, ls, lw_h, zorder, alpha=alpha, fill=name in fills)
                    if truth is not None:
                        ax.axvline(truth[jr], color=TRUTH_COLOR, lw=P["lw_truth"], ls=TDASH, zorder=100)
                    ax.set_xlim(*lims[r])
                    ax.set_ylim(bottom=0)
                    _despine_1d(ax)
                else:
                    for grp, name, s in arm_samples:
                        col, ls, lw_c, lw_h, zorder, alpha = styles[name]
                        _contour2d(ax, s[:, jc], s[:, jr], lims[c], lims[r], [nb[c], nb[r]],
                                   (smooth * nb[c] / bins, smooth * nb[r] / bins),
                                   col, ls, lw_c, zorder, alpha=alpha, fill=name in fills)
                    if truth is not None:
                        ax.axvline(truth[jc], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.8, ls=TDASH, alpha=0.8, zorder=150)
                        ax.axhline(truth[jr], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.8, ls=TDASH, alpha=0.8, zorder=150)
                        ax.plot(truth[jc], truth[jr], marker="s", ms=3.5, color=TRUTH_COLOR,
                                mec="white", mew=0.5, zorder=200)
                    ax.set_xlim(*lims[c])
                    ax.set_ylim(*lims[r])
                    _inner_ticks(ax.yaxis, lims[r], nt, tmarg)

                _inner_ticks(ax.xaxis, lims[c], nt, tmarg)

                if r == d - 1:
                    ax.set_xlabel(L(jc), labelpad=P["pad_label"])
                    _rotate(ax, P, "x")
                else:
                    ax.tick_params(labelbottom=False)

                if c == 0 and r > 0:
                    ax.set_ylabel(L(jr), labelpad=P["pad_label"])
                    _rotate(ax, P, "y")
                elif r != c:
                    ax.tick_params(labelleft=False)

        fig.align_ylabels([axes_main[r][0] for r in range(1, d)])
        fig.align_xlabels([axes_main[d - 1][c] for c in range(d)])

        if gs_zoom is not None:
            zx, zy = zoom_pair if zoom_pair is not None else (d - 2, d - 1)
            if zx > zy:
                zx, zy = zy, zx
            idx_x, idx_y = common_params[zx], common_params[zy]

            zoom_lims = []
            for idx, glim in ((idx_x, lims[zx]), (idx_y, lims[zy])):
                lo = min(np.percentile(s[:, idx], 0.5) for _, _, s in arm_samples)
                hi = max(np.percentile(s[:, idx], 99.5) for _, _, s in arm_samples)
                pad = 0.05 * (hi - lo) if hi > lo else 0.1
                zoom_lims.append((max(lo - pad, glim[0]), min(hi + pad, glim[1])))
            zl_x, zl_y = zoom_lims

            ax_z2d = fig.add_subplot(gs_zoom[1, 0])
            ax_zx = fig.add_subplot(gs_zoom[0, 0], sharex=ax_z2d)
            ax_zy = fig.add_subplot(gs_zoom[1, 1], sharey=ax_z2d)

            for grp, name, s in arm_samples:
                col, ls, lw_c, lw_h, zorder, alpha = styles[name]
                fl = name in fills
                _hist1d(ax_zx, s[:, idx_x], zl_x, bins, col, ls, lw_h, zorder, alpha=alpha, fill=fl)
                _hist1d(ax_zy, s[:, idx_y], zl_y, bins, col, ls, lw_h, zorder, alpha=alpha, horizontal=True, fill=fl)
                _contour2d(ax_z2d, s[:, idx_x], s[:, idx_y], zl_x, zl_y, bins, smooth,
                           col, ls, lw_c * (1.0 if AA else 1.1), zorder, alpha=alpha, fill=fl)

            if truth is not None:
                ax_zx.axvline(truth[idx_x], color=TRUTH_COLOR, lw=P["lw_truth"], ls=TDASH, zorder=100)
                ax_zy.axhline(truth[idx_y], color=TRUTH_COLOR, lw=P["lw_truth"], ls=TDASH, zorder=100)
                ax_z2d.axvline(truth[idx_x], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.8, ls=TDASH, alpha=0.8, zorder=150)
                ax_z2d.axhline(truth[idx_y], color=TRUTH_COLOR, lw=P["lw_truth"] * 0.8, ls=TDASH, alpha=0.8, zorder=150)
                ax_z2d.plot(truth[idx_x], truth[idx_y], marker="s", ms=6, color=TRUTH_COLOR,
                            mec="white", mew=1.0, zorder=200)

            ax_z2d.set_xlim(*zl_x); ax_z2d.set_ylim(*zl_y)
            ax_zx.set_ylim(bottom=0); ax_zy.set_xlim(left=0)
            _despine_1d(ax_zx)
            _despine_1d(ax_zy, horizontal=True)
            ax_zx.tick_params(labelbottom=False)
            ax_zy.tick_params(labelleft=False)

            _inner_ticks(ax_z2d.xaxis, zl_x, 4, 0.06)
            _inner_ticks(ax_z2d.yaxis, zl_y, 4, 0.06)
            ax_z2d.set_xlabel(L(idx_x), labelpad=P["pad_label"])
            ax_z2d.set_ylabel(L(idx_y), labelpad=P["pad_label"])
            _rotate(ax_z2d, P, "x"); _rotate(ax_z2d, P, "y")

            ax_m = axes_main[zy][zx]
            (x0, x1), (y0, y1) = zl_x, zl_y
            box_kw = dict(color="0.35", lw=P["lw_axes"], ls=(0, (3, 2)))
            ax_m.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, zorder=25, **box_kw))
            if P["name"] == "beamer":
                pairs = [((x1, y1), (0, 1)), ((x1, y0), (0, 0))]
            else:
                pairs = [((x0, y1), (0, 0)), ((x1, y0), (1, 0))]
            for xyA, xyB in pairs:
                fig.add_artist(ConnectionPatch(xyA=xyA, coordsA=ax_m.transData,
                                               xyB=xyB, coordsB=ax_z2d.transAxes,
                                               alpha=0.8, zorder=-1, **box_kw))

        handles = []
        for grp in unique_groups:
            col = group_colors[grp]
            handles.append(Line2D([], [], color=col, ls="-", lw=P["lw_contour"] * (1.0 if AA else 1.6), label=tex(_S._ps.arm_label(grp) if _S._ps is not None else grp, P)))

        if truth is not None:
            handles.append(Line2D([], [], color=TRUTH_COLOR, lw=P["lw_truth"], ls=TDASH, marker="s",
                                  ms=5, mec="white", mew=0.8, label="truth"))

        if AA:
            from matplotlib.patches import Patch as _Patch
            ents = []
            for grp in unique_groups:
                g = gst[grp]
                h = (_Patch(facecolor=_S.AA_REF_FILL, edgecolor=_S.AA_REF_EDGE, lw=_S._aa.LW["contour"])
                     if g["fill"] else Line2D([], [], color=g["color"], ls=g["ls"], lw=P["lw_contour"]))
                ents.append((g["header"] or g["short"], g["short"] if g["header"] else "", h))
            extra = [[("true value", handles[-1])]] if truth is not None else []
            H_, L_, nc, hi = _S.grouped_legend(ents, extra, P)
            _S.save_grouped_legend(H_, L_, nc, hi,
                                   _S.legend_path(re.sub(r"corner_sim\d+_", "corner_", out_path)), P,
                                   title=CONTOUR_LABEL)
        elif False:
            _S.save_legend(handles, [h.get_label() for h in handles],
                           _S.legend_path(re.sub(r"corner_sim\d+_", "corner_", out_path)), P,
                           ncol=4 if P.get("width") == "column" else 7, title=CONTOUR_LABEL)
        else:
          leg = fig.legend(handles=handles, loc=leg_loc, bbox_to_anchor=leg_anchor,
                         bbox_transform=fig.transFigure, frameon=False, borderaxespad=0.0,
                         handlelength=2.4, handletextpad=0.7, labelspacing=0.45,
                         title=tex(CONTOUR_LABEL, P), title_fontsize=P["fs_legend"] * 0.9)
          leg._legend_box.align = "left"

        if title and not AA:
            fig.text(0.01, 0.99, tex(title, P), ha="left", va="top",
                     fontsize=P["fs_title"], color="0.25")

        fig.savefig(out_path, **save_kwargs(P))
        if also_png:
            fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300, **save_kwargs(P))
        plt.close(fig)
    return out_path


def render_corner(arm_samples, truth, common_params, labels, out_pdf,
                  style=None, bins=30, smooth=1.0, title=None, also_png=False,
                  zoom_pair=None, fixed_lims=None):
    styles = resolve_styles(style)
    paths = []
    for st in styles:
        P = get_preset(st)
        path = with_suffix(out_pdf, st, len(styles))
        paths.append(_render_one(arm_samples, truth, common_params, labels, path, P,
                                 bins, smooth, title, also_png, zoom_pair, fixed_lims))
    return paths[0] if len(paths) == 1 else paths

def pick_corner_sim(arms, maps, common_params, thin, burnin, dlogp, n_corner):
    from eval import load_chain
    shared = set(maps[0])
    for m in maps[1:]:
        shared &= set(m)
    shared = np.array(sorted(shared))

    if len(shared) == 0:
        print("[corner] selected arms share no simulations", file=sys.stderr)
        return None

    ref, rmap = arms[0], maps[0]
    gvs = {}
    for s in shared:
        ld = load_chain(rmap[s], ref["n_params"], burnin, thin, dlogp)
        if ld is None:
            continue
        cov = np.cov(ld[0][:, common_params], rowvar=False)
        sign, ldet = np.linalg.slogdet(np.atleast_2d(cov))
        if sign > 0 and np.isfinite(ldet):
            gvs[int(s)] = np.exp(0.5 * ldet)

    if not gvs:
        return None

    med = np.median(list(gvs.values()))
    return sorted(gvs, key=lambda s: abs(gvs[s] - med))[:n_corner]

def plot_corner(report, out, burnin_frac, thin, dlogp, n_corner, common_params, labels,
                style=None, bins=30, smooth=1.0, title_sim=True):
    from eval import list_chains, chain_sim, load_chain

    for mode, arms in report.items():
        if not arms:
            continue

        maps = [{chain_sim(f): f for f in list_chains(a["dir"]) if chain_sim(f) is not None}
                for a in arms]
        sims_to_plot = pick_corner_sim(arms, maps, common_params, thin, burnin_frac, dlogp, n_corner)
        if not sims_to_plot:
            continue

        safe_mode = re.sub(r"[^A-Za-z0-9]+", "_", mode)
        jobs = []
        for sim in sims_to_plot:
            out_path = os.path.join(out, f"corner_sim{sim}_{safe_mode}.{FIGFMT}")

            arm_samples, truth = [], None
            for a, m in zip(arms, maps):
                if sim not in m:
                    continue
                ld = load_chain(m[sim], a["n_params"], burnin_frac, thin, dlogp)
                if ld is None:
                    continue
                s, tr = ld
                if truth is None:
                    truth = tr
                arm_samples.append((a.get("group", "default"), a["name"], s))

            if not arm_samples:
                continue

            unique_groups = list(dict.fromkeys(grp for grp, _, _ in arm_samples))
            for g in unique_groups:
                g_samples = [s for grp, name, s in arm_samples if grp == g]
                if len(g_samples) > 1:
                    pooled_chain = np.concatenate(g_samples, axis=0)
                    arm_samples.append((g, f"{g} (Pooled)", pooled_chain))

            jobs.append((sim, out_path, arm_samples, truth))

        per = [_drawn_limits(a, common_params) for _, _, a, _ in jobs]
        widths = [max(p[k][1] - p[k][0] for p in per) for k in range(len(common_params))]
        if FIXED_LIMITS and jobs:
            print("[corner] common axis widths: " + ", ".join(f"{w:.3f}" for w in widths))
        for (sim, out_path, arm_samples, truth), own in zip(jobs, per):
            if ZOOM_SET:
                zpath = os.path.splitext(out_path)[0] + "_zoom" + os.path.splitext(out_path)[1]
                written = render_corner(arm_samples, truth, common_params, labels, zpath,
                                        style=style, bins=bins, smooth=smooth,
                                        title=(f"sim {sim}" if title_sim else None),
                                        fixed_lims=own)
                for p in np.atleast_1d(written):
                    print(f"[corner] wrote {p}")
            fixed = None
            if FIXED_LIMITS:
                fixed = [(0.5 * (lo + hi) - 0.5 * w, 0.5 * (lo + hi) + 0.5 * w)
                         for (lo, hi), w in zip(own, widths)]
            written = render_corner(arm_samples, truth, common_params, labels, out_path,
                                    style=style, bins=bins, smooth=smooth,
                                    title=(f"sim {sim}" if title_sim else None),
                                    fixed_lims=fixed)
            for p in np.atleast_1d(written):
                print(f"[corner] wrote {p}")
