#!/usr/bin/env python3
"""aa_style.py -- the A&A figure style of the SBC calibration figures (sbc_full)."""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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

AX = {
    "label": 8.0,
    "tick": 7.0,
    "labelpad": 1.5,
    "tickpad": 2.0,
    "panel": 7.0,
    "legend": 6.2,
    "table": 5.6,
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

CONFIG = {
    "ref":      ((0.102, 0.102, 0.102), "o"),
    "A_k4":     ((0.902, 0.624, 0.000), "s"),
    "A_k8":     ((0.000, 0.620, 0.451), "s"),
    "A_k16":    ((0.800, 0.475, 0.655), "s"),
    "VMIM_100": ((0.000, 0.447, 0.698), "D"),
    "VMIM_50":  ((0.337, 0.706, 0.914), "D"),
    "VMIM_25":  ((0.200, 0.133, 0.533), "D"),
    "MSE_100":  ((0.835, 0.369, 0.000), "^"),
    "MSE_50":   ((0.941, 0.690, 0.000), "^"),
    "MSE_25":   ((0.533, 0.133, 0.333), "^"),
}

REGION = {
    "calibrated": (0.827, 0.910, 0.961),
    "marginal":   (0.894, 0.847, 0.941),
    "miscal":     (0.945, 0.898, 0.784),
    "strong":     (1.000, 1.000, 1.000),
}

TOY_COLOR = "0.45"
CROSS_DARKEN = 0.3

_CMASTRO_HEX = {
    "hesperia": (
        "00000002000d04001605001d07002308012809012c0a01300c01340d01370e023a0f023d"
        "10024011024212034413034614034815044a16044c17044e19044f1a05511b05521c0654"
        "1d06551e06571f075820075a22075b23085d24085e250960270961280a63290a642b0b65"
        "2c0b672d0c682f0c69300d6b320d6c330e6d340e6e360f6f370f713910723a11733c1174"
        "3d12753f127640137742147843147945157a46167b48167c49177d4b187d4c187e4e197f"
        "4f1a80511a80531b81541c82561d83571d83591e845a1f845c20855d20865f2186612287"
        "6223876423886524886725886926896a27896c288a6d288a6f298a702a8b722b8b742c8b"
        "752d8b772d8c782e8c7a2f8c7b308c7d318c7f328d80338d82348d83348d85358d86368d"
        "88378d89388d8b398d8c3a8d8e3b8d8f3c8d913d8d933e8d943f8d96408d97418c99428c"
        "9a438c9c438c9d448c9e458ca0468ba1478ba3488ba4498ba64a8aa74b8aa94c8aaa4d89"
        "ac4f89ad5089ae5188b05288b15388b35487b45587b65687b75786b85886ba5985bb5a85"
        "bc5b84be5c84bf5d83c15e83c25f82c36182c56281c66381c76480c9657fca667fcb677e"
        "cd687ece697dcf6b7cd06c7cd26d7bd36e7ad46f7ad57079d77178d87378d97477da7576"
        "dc7675dd7775de7874df7a73e07b72e27c72e37d71e37f70e48070e58270e6836fe6856f"
        "e7866ee8876ee8896ee98a6dea8c6dea8d6deb8f6ceb906cec926ced936ced956cee966b"
        "ee976bef996bef9a6bf09c6bf09d6bf19f6bf1a06bf2a26bf2a36bf3a56bf3a66bf4a86c"
        "f4a96cf5ab6cf5ac6cf5ad6df6af6df6b06df7b26ef7b36ef7b56ff8b66ff8b870f8b971"
        "f9bb71f9bc72f9be73fabf74fac174fac275fac376fbc577fbc679fbc87afbc97bfccb7c"
        "fccc7efcce7ffccf80fcd082fdd284fdd385fdd587fdd689fdd88bfdd98dfeda8ffedc91"
        "fedd94fedf96fee098fee19bfee39efee4a1fee6a3fee7a7ffe8aaffeaadffebb0ffecb4"
        "ffeeb7ffefbbfff0bffff1c3fff3c7fff4ccfff5d0fff6d5fff7d9fff9defffae3fffbe9"
        "fffceefffdf3fffef9ffffff"
    ),
    "lacerta": (
        "00000003000b05001406011a0801200901240a02280b022c0b032f0c04320d04350d0538"
        "0e063a0e063c0f073f0f0841100943100944110a46110b48110c49120d4b120e4c120f4e"
        "130f4f13105013115114125314135414145514155615165715175815185915195a161a5b"
        "161c5c161d5d161e5e171f5e17205f172160172261182361182462182662182763182864"
        "192964192a65192b65192c651a2e661a2f661a30671a31671b32671b33681b34681b3668"
        "1c37681c38691c39691c3a691d3b691d3d691d3e691d3f6a1e406a1e416a1e426a1e446a"
        "1f456a1f466a1f476a20486a20496a204a6a214c6a214d6a214e6a224f6a22506a22516a"
        "23536a23546a24556a24566a245769255869255969265b69265c69275d69275e69285f69"
        "2860682862682963682964682a65682a66672b67672c68672c6a672d6b672d6c662e6d66"
        "2e6e662f6f66307065307265317365327465327564337664347764347864357a63367b63"
        "367c63377d62387e62397f623980613a82613b83613c84603d85603d86603e875f3f885f"
        "40895f418b5e428c5e438d5e448e5d458f5d46905c47915c48925c49945b4a955b4b965a"
        "4c975a4d98594e99594f9a59509b58519d58529e57539f5755a05656a15657a25558a355"
        "5aa4545ba5545ca6535da8535fa95260aa5261ab5163ac5164ad5066ae5067af4f68b04f"
        "6ab14e6bb24d6db34d6eb44c70b64c72b74b73b84b75b94a76ba4978bb497abc487bbd47"
        "7dbe477fbf4681c04582c14584c24486c34388c4438ac5428cc6418ec7418fc84091c93f"
        "93ca3f95cb3e98cc3e9acc3f9dcd41a0ce42a2cf44a5cf46a8d048aad149add24bb0d24d"
        "b2d34fb5d451b7d554bad556bcd658bfd75ac1d75dc3d85fc6d962c8da64cada67cddb69"
        "cfdc6cd1dd6fd3dd72d5de74d8df77dae07adce07ddee181dfe284e1e387e3e38ae5e48e"
        "e7e591e8e694eae798ece89cede89fefe9a3f0eaa7f1ebaaf3ecaef4edb2f5eeb6f6efba"
        "f7efbef8f0c2f9f1c6faf2cafbf3cffbf4d3fcf5d7fdf6dbfdf7e0fef8e4fef9e9fefbed"
        "fffcf1fffdf6fffefaffffff"
    ),
    "laguna": (
        "0000000301030602060902090c030d0f040f11041213051515061717061a19071c1b071e"
        "1c08201e08221f0924200926220a28230a2a240a2c250b2d260b2f270c31280c33290d34"
        "2a0d362b0e382c0e3a2d0f3c2e0f3d2e103f2f104130114331114532124733124933134b"
        "34144d35144e36155036165237165438175638185839185a3a195c3a1a5e3b1a603b1b62"
        "3c1c643c1d663d1e683d1e6a3e1f6c3e206e3f21703f22723f237440247640247840257a"
        "41267b41277d41287f412981422a83422b85422c86422d88422e8a422f8c43308d43318f"
        "43339143349243359443369643379743389943399a423a9c423c9d423d9f423ea0423fa2"
        "4240a34242a44143a54144a74145a84147a94148aa4049ab404aad404cae404daf3f4eb0"
        "3f50b13f51b23e52b33e53b33e55b43d56b53d57b63d59b73c5ab73c5bb83c5db93b5eb9"
        "3b5fba3a61bb3a62bb3a63bc3965bc3966bd3968bd3869bd386abe376cbe376dbe376ebf"
        "3670bf3671bf3572bf3574c03575c03476c03478c03479c0337bc0337cc0337dc0327fc0"
        "3280c03281c03183c03184c03185c03087c03088c0308ac0308bbf2f8cbf2f8ebf2f8fbf"
        "2f90bf2f92be2f93be2f94bd3096bd3097bc3098bc319abb319bbb329cba329eba339fb9"
        "33a0b934a2b835a3b835a4b736a6b737a7b638a8b638a9b539abb53aacb43badb43caeb3"
        "3db0b33eb1b33fb2b240b3b241b5b142b6b144b7b145b8b046bab047bbb049bcb04abdaf"
        "4bbeaf4dc0af4ec1af50c2ae52c3ae53c4ae55c5ae56c7ae58c8ae5ac9ae5ccaae5ecbae"
        "60ccae61cdae63ceae65cfae67d0af6ad1af6cd3af6ed4af70d5b072d6b074d7b177d8b1"
        "79d9b17bdab27edbb380dcb382dcb485ddb487deb58adfb68ce0b78fe1b891e2b894e3b9"
        "97e4ba99e5bb9ce5bc9fe6bda1e7bfa4e8c0a7e9c1a9e9c2aceac4afebc5b2ecc6b4edc8"
        "b7edc9baeecbbdefccbfefcec2f0cfc5f1d1c8f2d3caf2d4cdf3d6d0f4d8d3f4dad5f5dc"
        "d8f6dedbf6e0ddf7e2e0f8e4e3f8e6e5f9e8e8f9eaebfaecedfbeff0fbf1f2fcf3f5fdf5"
        "f8fdf8fafefafdfefdffffff"
    ),
}
CMASTRO_SEQUENTIAL = ("laguna", "hesperia", "lacerta")
CMASTRO_DIVERGING = ("emph", "unph")


def _hex_rgb(s):
    return np.array([[int(s[i + j:i + j + 2], 16) / 255 for j in (0, 2, 4)]
                     for i in range(0, len(s), 6)])


CMASTRO_RGB = {k: _hex_rgb(v) for k, v in _CMASTRO_HEX.items()}

_w = CMASTRO_RGB["hesperia"][50:-12]
_c = CMASTRO_RGB["laguna"][50:-12]
CMASTRO_RGB["emph"] = np.concatenate((_w[::-1], [0.5 * (_w[0] + _c[0])], _c))
CMASTRO_RGB["unph"] = np.concatenate((_w, [0.5 * (_w[-1] + _c[-1])], _c[::-1]))

PASTEL = ["#f7f0da", "#e4d9ee", "#a9cfe9", "#4f8cc6", "#1f3b73"]

CMAPS = {}


def _register(cm):
    CMAPS[cm.name] = cm
    if cm.name not in mpl.colormaps:
        mpl.colormaps.register(cm)


for _name, _rgb in CMASTRO_RGB.items():
    _register(ListedColormap(_rgb, name=f"cma:{_name}"))
    _register(ListedColormap(_rgb[::-1], name=f"cma:{_name}_r"))


def luminance(c):
    """Relative brightness 0 (black) .."""
    r, g, b = mpl.colors.to_rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_on(c, dark=INK, light="white"):
    """Text colour readable on a background colour c (e.g. a heatmap cell)."""
    return light if luminance(c) < 0.5 else dark


def cmap(name="laguna", light=0.0, dark=0.12):
    """Sequential map for "more = darker" data (r_I tables, densities): a cmastro map ("laguna" ..."""
    if name == "pastel":
        return LinearSegmentedColormap.from_list("aa:pastel", PASTEL)
    if name not in CMASTRO_SEQUENTIAL:
        raise ValueError(f"cmap(): {name!r} not in {CMASTRO_SEQUENTIAL + ('pastel',)}")
    rgb = CMASTRO_RGB[name][::-1]
    n = len(rgb)
    rgb = rgb[int(round(light * n)): n - int(round(dark * n))]
    tag = "" if (light, dark) == (0.0, 0.12) else f"[{light:g},{dark:g}]"
    return ListedColormap(rgb, name=f"aa:{name}{tag}")


for _name in CMASTRO_SEQUENTIAL + ("pastel",):
    _register(cmap(_name))
del _name, _rgb, _w, _c

try:
    import cmasher as cmr                    # noqa: F401  (registers "cmr.*" maps)
except ImportError:
    cmr = None
try:
    import palettable
    from palettable.palette import Palette as _Palette
except ImportError:
    palettable = None


def _palettable(path):
    if palettable is None:
        raise ImportError("palettable is not installed: python -m pip install palettable")
    mod, _, attr = path.removeprefix("palettable.").rpartition(".")
    return getattr(importlib.import_module(f"palettable.{mod}"), attr)


def get_cmap(name):
    """Any colour map by name: "aa:laguna", "cma:hesperia", "cmr.rainforest", "viridis", or a palettable ..."""
    if name in mpl.colormaps:
        return mpl.colormaps[name]
    if "." in name and not name.startswith("cmr."):
        return _palettable(name).mpl_colormap
    hint = "  (CMasher not installed?)" if name.startswith("cmr.") and cmr is None else ""
    raise ValueError(f"unknown colour map {name!r}{hint}")


def colors(name, n=None, lo=0.0, hi=1.0):
    """Discrete hex colours."""
    if n is None and name not in mpl.colormaps and "." in name and not name.startswith("cmr."):
        return list(_palettable(name).hex_colors)
    cm = get_cmap(name)
    return [mpl.colors.to_hex(cm(x)) for x in np.linspace(lo, hi, n or 8)]


def catalogue():
    """Every map and palette reachable from here, grouped for the reference sheet: [(title, how to import ..."""
    out = [
        ("aa_style: heatmap maps, light -> dark (black end trimmed)",
         'import aa_style as aa   ->   cmap="aa:<name>"   or   aa.cmap("<name>")',
         [(f"aa:{n}", CMAPS[f"aa:{n}"]) for n in CMASTRO_SEQUENTIAL + ("pastel",)]),
        ("cmastro (embedded in aa_style, no install)",
         'import aa_style   ->   cmap="cma:<name>"   (+ "cma:<name>_r")',
         [(f"cma:{n}", CMAPS[f"cma:{n}"]) for n in CMASTRO_SEQUENTIAL + CMASTRO_DIVERGING]),
    ]
    if cmr is not None:
        groups = {}
        for nm in sorted(n for n in mpl.colormaps if n.startswith("cmr.") and not n.endswith("_r")):
            groups.setdefault(cmr.get_cmap_type(nm), []).append((nm, mpl.colormaps[nm]))
        for t in sorted(groups, key=lambda t: ["sequential", "diverging", "cyclic"].index(t)
                        if t in ("sequential", "diverging", "cyclic") else 9):
            out.append((f"CMasher, {t}",
                        'import cmasher as cmr   ->   cmap="cmr.<name>"   or   cmr.<name>',
                        groups[t]))
    if palettable is not None:
        for m in pkgutil.walk_packages(palettable.__path__, "palettable."):
            if "test" in m.name:
                continue
            mod = importlib.import_module(m.name)
            best = {}
            for k, v in vars(mod).items():
                if not isinstance(v, _Palette) or k.endswith("_r"):
                    continue
                base, _, num = k.rpartition("_")
                base = base if num.isdigit() else k
                if base not in best or v.number > best[base][1].number:
                    best[base] = (k, v)
            if best:
                items = [(k, v.mpl_colors) for k, v in sorted(best.values(),
                                                               key=lambda kv: kv[0].lower())]
                out.append((m.name, f"from {m.name} import <name>", items))
    return out


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
        "image.cmap": "aa:laguna",
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


def _demo_cmaps():
    use()
    rows = [(n, CMAPS[n]) for n in CMAPS if not n.endswith("_r")]
    W, rh, left = WIDTH_MM["column"], 4.2, 21.0
    H = 1.5 + len(rows) * rh
    fig = plt.figure(figsize=(W * MM, H * MM))
    grad = np.linspace(0, 1, 256)[None, :]
    for i, (name, cm) in enumerate(rows):
        ax = fig.add_axes([left / W, 1 - (1.0 + (i + 0.85) * rh) / H, (W - left - 1) / W,
                           0.7 * rh / H])
        ax.imshow(grad, aspect="auto", cmap=cm)
        ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
        ax.tick_params(which="both", top=False, right=False)
        fig.text((left - 1.2) / W, 1 - (1.0 + (i + 0.5) * rh) / H, name, ha="right",
                 va="center", fontsize=FS["legend"], family="monospace", color=INK)
    save(fig, "aa_style_cmaps")


def _demo_palettes(name="aa_style_palettes"):
    from matplotlib.backends.backend_pdf import PdfPages
    use()
    cat = catalogue()
    W, H = 194.0, 277.0
    top, bottom, cgap = 17.0, 6.0, 6.0
    colw = (W - cgap) / 2
    row, sw = 3.55, 30.0
    head = 2 * row + 1.2
    pages, page, col, y = [], [], 0, top

    def new_column():
        nonlocal page, col, y
        col, y = col + 1, top
        if col == 2:
            pages.append(page); page, col = [], 0

    for title, how, items in cat:
        if y + head + row > H - bottom:
            new_column()
        page.append(("head", col, y, title, how)); y += head
        for nm, obj in items:
            if y + row > H - bottom:
                new_column()
                page.append(("head", col, y, title + "  (cont.)", how)); y += head
            page.append(("item", col, y, nm, obj)); y += row
        y += 1.4
    pages.append(page)

    n_items = sum(len(it) for _, _, it in cat)
    outdir = Path(".")
    with PdfPages(outdir / f"{name}.pdf") as pdf:
        for p, entries in enumerate(pages):
            fig = plt.figure(figsize=(W * MM, H * MM))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
            if p == 0:
                ax.text(0, 4.5, "Colour maps and palettes in aa_style.py", fontsize=10,
                        weight="bold", va="top", color=INK)
                ax.text(0, 9.2, f"{n_items} maps and palettes.  Any of them through aa_style:"
                        "  aa.get_cmap(name),  aa.colors(name, n).", fontsize=6.2, va="top",
                        color=MUTED)
                ax.text(0, 12.0, "palettable: largest size shown, '_N' = number of colours, "
                        "'_r' reversed;  use  <name>.mpl_colormap  (continuous),  "
                        "<name>.mpl_colors  /  <name>.hex_colors  (list).", fontsize=6.2,
                        va="top", color=MUTED)
            ax.text(W, H - 1.5, f"{p + 1} / {len(pages)}", ha="right", va="bottom",
                    fontsize=6, color=MUTED)
            for e in entries:
                kind, c, y = e[:3]
                x = c * (colw + cgap)
                if kind == "head":
                    title, how = e[3:]
                    ax.text(x, y + 0.6 * row, title, fontsize=7.2, weight="bold",
                            va="center", color=INK)
                    ax.text(x, y + 1.55 * row, how, fontsize=4.7, family="monospace",
                            va="center", color=MUTED)
                    continue
                nm, obj = e[3:]
                y0, h = y + 0.45, row - 0.9
                if isinstance(obj, mpl.colors.Colormap):
                    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=obj, aspect="auto",
                              extent=(x, x + sw, y0 + h, y0), interpolation="bilinear")
                else:
                    k = len(obj)
                    for i, rgb in enumerate(obj):
                        ax.add_patch(Rectangle((x + i * sw / k, y0), sw / k, h, fc=rgb,
                                               ec="none"))
                ax.add_patch(Rectangle((x, y0), sw, h, fc="none", ec=INK, lw=0.3))
                ax.text(x + sw + 1.5, y0 + h / 2, nm, fontsize=5.6, family="monospace",
                        va="center", color=INK)
            pdf.savefig(fig)
            fig.savefig(outdir / f"{name}_p{p + 1}.png", dpi=200)
            plt.close(fig)
    print(f"wrote {name}.pdf  ({len(pages)} pages, {n_items} maps and palettes)")


if __name__ == "__main__":
    {"cmaps": _demo_cmaps, "palettes": _demo_palettes}.get(
        (sys.argv[1:2] or [""])[0], _demo)()
