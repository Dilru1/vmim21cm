import os
import shutil

VIVID = ["#0077BB", "#EE3377", "#009988", "#CC3311", "#DDAA33", "#33BBEE"]
TRUTH_COLOR = "#222222"

try:
    from _palette import OKABE_ITO, arm_linestyle, FIGFMT
except ImportError:
    OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    def arm_linestyle(i): return "-"
    FIGFMT = "pdf"

PALETTES = {"vivid": VIVID, "okabe": OKABE_ITO}

STYLE = "both"

import sys as _sys
_PP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_plots")
PAPER_WIDTH = os.environ.get("PAPER_FIGWIDTH", "column")
try:
    if os.path.isdir(_PP) and _PP not in _sys.path:
        _sys.path.insert(0, os.path.abspath(_PP))
    import paper_style as _ps
except ImportError:
    _ps = None
if _ps is not None:
    VIVID = list(_ps.CYCLE[:-1])
    TRUTH_COLOR = _ps.INK
    PALETTES = {"vivid": VIVID, "okabe": OKABE_ITO}


def group_color_map(groups):
    """{group name: colour}."""
    groups = list(groups)
    if _ps is not None:
        return dict(zip(groups, _ps.arm_colors(groups)))
    return {g: VIVID[i % len(VIVID)] for i, g in enumerate(groups)}

CONTOUR_LEVEL = 0.865

SHOW_SEEDS = os.environ.get("PLOT_SHOW_SEEDS", "1") != "0"
LW_SCALE = float(os.environ.get("PLOT_LW_SCALE", "1.0"))
CONTOUR_LABEL = r"95% credible region"

PRESETS = {
    "beamer": dict(
        figsize=(8.0, 4.5),
        tight=False,
        family="sans-serif", fonts=["DejaVu Sans", "Helvetica", "Arial"],
        usetex=False, mathtext="dejavusans",
        fs_label=13, fs_tick=10.5, fs_legend=11, fs_title=12,
        lw_axes=1.1, lw_contour=1.8, lw_hist=1.6, lw_truth=1.3, lw_ref=1.4,
        tick_major=3.5, tick_minor=2.0,
        pad_label=3.5, panel_space=0.05,
        corner_ticks=3, rotate_ticks=0,
    ),

    "paper": dict(
        fig_width=7.0, sbc_panel_height=1.15,
        tight=True,
        family="serif",
        fonts=["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        usetex=True, mathtext="cm",
        fs_label=11, fs_tick=9, fs_legend=9.5, fs_title=10.5,
        lw_axes=0.8, lw_contour=1.3, lw_hist=1.3, lw_truth=1.0, lw_ref=1.1,
        tick_major=3.5, tick_minor=1.8,
        pad_label=3.5, panel_space=0.05,
        corner_ticks=3, rotate_ticks=0,
    ),
}


if _ps is not None:
    PRESETS["paper"] = dict(_ps.validation_preset(PAPER_WIDTH), bridged=True,
                            legend_ncol=2 if PAPER_WIDTH == "column" else 4)


SBC_PAPER_WIDTHS = {"1col": ("column", "stack"), "2col": ("full", "row")}


def sbc_paper_presets():
    """[(suffix, preset)] for the paper-style SBC figure; [] without paper_style."""
    if _ps is None:
        return []
    out = []
    for tag, (width, layout) in SBC_PAPER_WIDTHS.items():
        P = dict(_ps.validation_preset(width), bridged=True, name="paper",
                 legend_ncol=2 if width == "column" else 4, sbc_layout=layout)
        out.append((f"paper_{tag}", P))
    return out


def resolve_styles(style=None):
    """None -> STYLE; 'both' -> ['beamer', 'paper']; str -> [str]; list -> list."""
    style = STYLE if style is None else style
    if isinstance(style, str):
        style = ["beamer", "paper"] if style == "both" else [style]
    for s in style:
        if s not in PRESETS:
            raise ValueError(f"unknown style {s!r}; choose from {list(PRESETS)} or 'both'")
    return list(style)


_LATEX_OK = None
def latex_available():
    global _LATEX_OK
    """True only if matplotlib can actually typeset a test string with LaTeX."""
    if _LATEX_OK is None:
        _LATEX_OK = False
        if shutil.which("latex") and shutil.which("kpsewhich"):
            try:
                import matplotlib
                from matplotlib.texmanager import TexManager
                with matplotlib.rc_context({"text.usetex": True}):
                    TexManager.get_text_width_height_descent(r"$\sigma$ 1", 10)
                _LATEX_OK = True
            except Exception:
                _LATEX_OK = False
    return _LATEX_OK


def get_preset(style):
    P = dict(PRESETS[style])
    P["name"] = style
    for k in ("lw_contour", "lw_hist", "lw_ref", "lw_truth"):
        if k in P:
            P[k] = P[k] * LW_SCALE
    if P.get("bridged"):
        return P
    if P.get("usetex") and not latex_available():
        print(f"[settings] LaTeX not found -> '{style}' falls back to mathtext")
        P["usetex"] = False
    return P


def tex(s, P):
    """Make a plain label safe for usetex: escape % & # _ outside $...$ math."""
    if not P.get("usetex") or not isinstance(s, str):
        return s
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        p = parts[i]
        for ch in "%&#_":
            p = p.replace("\\" + ch, ch).replace(ch, "\\" + ch)
        parts[i] = p
    return "$".join(parts)


def with_suffix(path, style, n_styles):
    """Append _<style> to the file name when several styles are rendered."""
    if n_styles <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{style}{ext}"


def save_kwargs(P):
    return dict(bbox_inches="tight", pad_inches=0.03) if P["tight"] else {}


def get_rc(P):
    if P.get("bridged") and _ps is not None:
        d = _ps.rc(P.get("usetex", False))
        d.update({"savefig.dpi": 300})
        return d
    serif = P["family"] == "serif"
    rc_dict = {
        "font.family": P["family"],
        ("font.serif" if serif else "font.sans-serif"): P["fonts"],
        "mathtext.fontset": P.get("mathtext", "dejavusans"),
        "text.usetex": P.get("usetex", False),
        "font.size": P["fs_tick"],
        "axes.labelsize": P["fs_label"],
        "axes.titlesize": P["fs_title"],
        "xtick.labelsize": P["fs_tick"],
        "ytick.labelsize": P["fs_tick"],
        "legend.fontsize": P["fs_legend"],
        "axes.linewidth": P["lw_axes"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": P["tick_major"], "ytick.major.size": P["tick_major"],
        "xtick.minor.size": P["tick_minor"], "ytick.minor.size": P["tick_minor"],
        "xtick.major.width": P["lw_axes"], "ytick.major.width": P["lw_axes"],
        "xtick.minor.width": P["lw_axes"] * 0.75,
        "ytick.minor.width": P["lw_axes"] * 0.75,
        "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
    if P.get("usetex", False):
        rc_dict["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
    return rc_dict

try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "aa_style", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aa_style.py"))
    _aa = _ilu.module_from_spec(_spec)
    _sys.modules["aa_style"] = _aa
    _spec.loader.exec_module(_aa)
except (ImportError, FileNotFoundError):
    _aa = None
USE_AA = _aa is not None and os.environ.get("PLOT_AA", "1") != "0"

AA_LW_DATA = 0.6
AA_LW_SEED = 0.5
AA_LW_SBC = 0.6
AA_LABELS = [r"$\log_{10} f_{\rm X}$", r"$\tau$", r"$r_{\rm H/S}$",
             r"$\log_{10} M_{\rm min}$", r"$f_{\rm esc}$"]

import re as _re
AA_GROUP_KEYS = [
    (r".*(pdf_?ps|reference|baseline).*", "ref"),
    (r".*(patha|mlp).*t(4)\b.*|.*patha_t4.*", "A_k4"),
    (r".*patha_t8.*|.*mlp.*t8.*", "A_k8"),
    (r".*patha_t16.*|.*mlp.*t16.*", "A_k16"),
    (r".*vmim.*(50h|nsqrt2).*", "VMIM_50"),
    (r".*vmim.*(25h|n2).*", "VMIM_25"),
    (r".*vmim.*", "VMIM_100"),
    (r".*mse.*(50h|nsqrt2).*", "MSE_50"),
    (r".*mse.*(25h|n2).*", "MSE_25"),
    (r".*mse.*", "MSE_100"),
]


def aa_key(group):
    g = str(group).lower()
    for pat, key in AA_GROUP_KEYS:
        if _re.fullmatch(pat, g):
            return key
    return None


def aa_preset(width):
    """Paper preset built from aa_style.py."""
    a = _aa
    return dict(
        aa=True, name="paper",
        fig_width=a.WIDTH_MM[width] * a.MM, width=width,
        sbc_panel_height=0.62 if width == "column" else 0.85,
        tight=True, family="serif", fonts=list(a.FONT_SERIF), usetex=False,
        mathtext=a.MATHTEXT,
        fs_label=a.FS["label"], fs_tick=a.FS["tick"], fs_legend=a.FS["legend"],
        fs_title=a.FS["panel"], fs_panel=a.FS["panel"],
        lw_axes=a.LW["axes"], lw_contour=AA_LW_DATA, lw_hist=AA_LW_DATA,
        lw_seed=AA_LW_SEED, lw_truth=a.LW["guide"], lw_ref=a.LW["guide"],
        dash=a.DASH, ink=a.INK,
        tick_major=a.TICK["major"], tick_minor=a.TICK["minor"],
        pad_label=2.0, panel_space=0.06, corner_ticks=3, rotate_ticks=0,
        labels=AA_LABELS,
    )


def _scale_lw(P):
    for k in ("lw_contour", "lw_hist", "lw_ref", "lw_truth", "lw_seed"):
        if k in P:
            P[k] = P[k] * LW_SCALE
    return P


def save_legend(handles, labels, path, P, ncol=None, width=None, title=None):
    """Legend alone, as its own figure (tight bbox)."""
    import matplotlib.pyplot as plt
    width = width or P.get("width", "column")
    if ncol is None:
        ncol = 2 if width == "column" else 4
    with plt.rc_context(get_rc(P)):
        fig = plt.figure(figsize=(P["fig_width"], 1.0))
        fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
                   handlelength=2.0, handletextpad=0.5, columnspacing=1.2,
                   labelspacing=0.35, borderaxespad=0.0, title=title,
                   title_fontsize=P["fs_legend"])
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"[legend] wrote {path}")
    return path


def legend_path(path):
    root, ext = os.path.splitext(path)
    return f"{root}_legend{ext}"


_get_preset_old, _sbc_paper_presets_old, _get_rc_old, _group_color_map_old = \
    get_preset, sbc_paper_presets, get_rc, group_color_map

if USE_AA:
    def get_preset(style):
        if style == "paper":
            return _scale_lw(aa_preset(PAPER_WIDTH))
        return _get_preset_old(style)

    def sbc_paper_presets():
        out = []
        for tag, (width, layout) in SBC_PAPER_WIDTHS.items():
            P = _scale_lw(aa_preset(width))
            P.update(sbc_layout=layout, legend_ncol=2 if width == "column" else 4)
            out.append((f"paper_{tag}", P))
        return out

    def get_rc(P):
        if P.get("aa"):
            d = _aa.rc()
            d.update({"savefig.dpi": 300, "lines.solid_capstyle": "butt"})
            return d
        return _get_rc_old(P)

    def group_color_map(groups):
        groups = list(groups)
        base = _group_color_map_old(groups)
        return {g: (_aa.CONFIG[aa_key(g)][0] if aa_key(g) else base[g]) for g in groups}


AA_REF_ALPHA = 0.45
AA_REF_FILL = (0.80, 0.80, 0.80, AA_REF_ALPHA)
AA_REF_EDGE = "0.45"
AA_HEADERS = {"ref": ("Reference", "PDF+PS"),
              "A_k4": ("Path A", r"$k=4$"), "A_k8": ("Path A", r"$k=8$"),
              "A_k16": ("Path A", r"$k=16$"),
              "VMIM_100": ("Path B, VMIM", r"100$\,$h"), "VMIM_50": ("Path B, VMIM", r"50$\,$h"),
              "VMIM_25": ("Path B, VMIM", r"25$\,$h"),
              "MSE_100": ("Path B, MSE", r"100$\,$h"), "MSE_50": ("Path B, MSE", r"50$\,$h"),
              "MSE_25": ("Path B, MSE", r"25$\,$h")}


def aa_group_style(group, color):
    """-> dict(color, ls, fill, header, short) for one list/group."""
    k = aa_key(group) if USE_AA else None
    ls = "-"
    if k and k.endswith("_50"):
        ls = _aa.DASH
    elif k and k.endswith("_25"):
        ls = (0, (1, 1.5))
    hdr, short = AA_HEADERS.get(k, (None, None))
    return dict(color=color, ls=ls, fill=(k == "ref"), header=hdr,
                short=short or (_ps.arm_label(group) if _ps is not None else group))


def grouped_legend(entries, extra_cols=(), P=None):
    """entries: [(header, short, handle)] in plot order -> (handles, labels, ncol, header_idx) laid out ..."""
    from matplotlib.lines import Line2D
    cols = {}
    for hdr, short, h in entries:
        cols.setdefault(hdr, []).append((short, h))
    blocks = [[(hdr, None)] + items for hdr, items in cols.items()]
    blocks += [[(None, None)] + list(c) for c in extra_cols]
    nrow = max(len(b) for b in blocks)
    H, L, hdr_idx = [], [], []
    for b in blocks:
        b = b + [("", None)] * (nrow - len(b))
        for i, (lab, h) in enumerate(b):
            if h is None:
                if i == 0 and lab:
                    hdr_idx.append(len(L))
                h = Line2D([], [], color="none")
            H.append(h); L.append(lab or "")
    return H, L, len(blocks), hdr_idx


def save_grouped_legend(H, L, ncol, hdr_idx, path, P, title=None):
    import matplotlib.pyplot as plt
    with plt.rc_context(get_rc(P)):
        fig = plt.figure(figsize=(P["fig_width"], 1.0))
        leg = fig.legend(H, L, loc="center", ncol=ncol, frameon=False,
                         handlelength=2.2, handletextpad=0.5, columnspacing=1.4,
                         labelspacing=0.3, borderaxespad=0.0, title=title,
                         title_fontsize=P["fs_legend"])
        for i, t in enumerate(leg.get_texts()):
            if i in hdr_idx:
                t.set_fontstyle("italic")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"[legend] wrote {path}")
    return path


def nice_ticks(lo, hi, n=3, margin=0.08, nmax=None):
    """Exactly n (else as close as possible) round ticks inside [lo, hi], kept `margin` x span away from ..."""
    import numpy as np
    span = hi - lo
    a, b = lo + margin * span, hi - margin * span
    best = None
    for e in range(-4, 4):
        for m, nmin in ((1, 5), (2, 4), (2.5, 5), (5, 5)):
            st = m * 10.0 ** e
            t = np.arange(np.ceil(a / st - 1e-9) * st, b + 1e-9 * st, st)
            t = [round(v, 10) for v in t]
            if not t or (nmax is not None and len(t) > nmax):
                continue
            score = (abs(len(t) - n), 0 if len(t) >= n else 1)
            if best is None or score <= best[0]:
                best = (score, t, st / nmin)
    return best[1], best[2]
