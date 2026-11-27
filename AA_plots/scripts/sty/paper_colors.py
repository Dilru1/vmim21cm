#!/usr/bin/env python3
"""paper_colors.py -- ONE colour scheme for every figure of the paper."""
import os
import sys

import numpy as np
import matplotlib as mpl
import cmasher as cmr
try:
    import cmastro  # noqa: F401             # "cma:<name>"; fails on matplotlib >= 3.9
except Exception:
    cmastro = None


def _at(cmap, x):
    return mpl.colors.to_hex(mpl.colormaps[cmap](float(x)))


INK = "#1A1A1A"
MUTED = "#6B6B6B"
GUIDE = "#9E9E9E"
WHITE = "#FFFFFF"

ARM_FAMILY = {
    "ref":  (None,          "Reference",    "o"),
    "A":    ("cmr.emerald", "Path A",       "s"),
    "VMIM": ("cmr.ocean",   "Path B, VMIM", "D"),
    "MSE":  ("cmr.ember",   "Path B, MSE",  "^"),
}
ARM_POS = {
    "ref":      ("ref",  None, "PDF+PS (reference)"),
    "A_k4":     ("A",    0.40, r"Path A, $k=4$"),
    "A_k8":     ("A",    0.60, r"Path A, $k=8$"),
    "A_k16":    ("A",    0.80, r"Path A, $k=16$"),
    "VMIM_100": ("VMIM", 0.25, "VMIM 100 h"),
    "VMIM_50":  ("VMIM", 0.70, "VMIM 50 h"),
    "VMIM_25":  ("VMIM", 0.90, "VMIM 25 h"),
    "MSE_100":  ("MSE",  0.35, "MSE 100 h"),
    "MSE_50":   ("MSE",  0.70, "MSE 50 h"),
    "MSE_25":   ("MSE",  0.90, "MSE 25 h"),
}
REF_COLOR = INK
REF_FILL = "#A8A8A8"

ARMS = {}
for _k, (_fam, _x, _lab) in ARM_POS.items():
    _cm, _, _mk = ARM_FAMILY[_fam]
    ARMS[_k] = (_lab, REF_COLOR if _cm is None else _at(_cm, _x), _mk)
ARMS["ref_gmm"] = ("PDF+PS, GMM", "#8C8C8C", "v")
MAIN_ARMS = ["ref", "A_k8", "VMIM_100", "VMIM_50", "MSE_100", "MSE_50"]

NOISE_HOURS = (100, 50, 25)
NOISE_LS = {100: "-", 50: (0, (3, 2)), 25: (0, (1, 1.5))}
NOISE_GREYS = {100: "#3A3A3A", 50: "#6E6E6E", 25: "#A3A3A3"}

SEQ_NAME = "aa.iceburn_mid"
def _iceburn_mid():
    ice = mpl.colormaps["cmr.iceburn"]
    stops = [(0.00, ice(0.00)), (0.15, ice(0.10)), (0.32, ice(0.22)), (0.50, "#7B3F98"),
             (0.68, ice(0.74)), (0.84, ice(0.85)), (1.00, ice(1.00))]
    cm = mpl.colors.LinearSegmentedColormap.from_list("aa.iceburn_mid", stops, N=256)
    if "aa.iceburn_mid" not in mpl.colormaps:
        mpl.colormaps.register(cm, name="aa.iceburn_mid")


_iceburn_mid()
for _nm, _src, _lo, _hi in (("aa.iceburn_warm", "cmr.iceburn_r", 0.0, 0.5),
                            ("aa.iceburn_cold", "cmr.iceburn", 0.0, 0.5)):
    if _nm not in mpl.colormaps:
        _cm = cmr.get_sub_cmap(_src, _lo, _hi); _cm.name = _nm
        mpl.colormaps.register(_cm, name=_nm)
SEQ_NAME = os.environ.get("AA_SEQ", SEQ_NAME)
SEQ_RANGE = (0.0, 1.0)
if os.environ.get("AA_SEQ_RANGE"):
    SEQ_RANGE = tuple(float(v) for v in os.environ["AA_SEQ_RANGE"].split(","))
SEQ_CMAP = cmr.get_sub_cmap(SEQ_NAME, *SEQ_RANGE)
SEQ_CMAP.name = "paper:seq"

Z_ORDER = ("12.06", "10.32", "8.18")
Z_POS = (0.92, 0.70, 0.48)
Z_ICE_POS = (0.20, 0.70, 0.88)
Z_COLORS = {z: mpl.colors.to_hex(mpl.colormaps["cmr.iceburn"](x)) for z, x in zip(Z_ORDER, Z_ICE_POS)}
Z_MARKERS = {"12.06": "o", "10.32": "s", "8.18": "^"}

DIV_NAME = "cmr.iceburn"
DIV_CMAP = mpl.colormaps[DIV_NAME]

REGIONS = {
    "calibrated": "#D6D6D6",
    "marginal":   "#E6E6E6",
    "miscal":     "#F3F3F3",
    "strong":     WHITE,
}
REGIONS_LIGHT = {
    "calibrated": "#E3E3E3",
    "marginal":   "#EEEEEE",
    "miscal":     "#F7F7F7",
    "strong":     WHITE,
}
def _on_white(c, alpha):
    r = np.array(mpl.colors.to_rgb(c))
    return mpl.colors.to_hex(1 - alpha * (1 - r))


REGION_HUE = mpl.colors.to_hex(mpl.colormaps["cmr.chroma"](0.60))
REGION_ALPHAS = (0.42, 0.22, 0.09)
REGIONS_SEQ = {
    "calibrated": _on_white(REGION_HUE, REGION_ALPHAS[0]),
    "marginal":   _on_white(REGION_HUE, REGION_ALPHAS[1]),
    "miscal":     _on_white(REGION_HUE, REGION_ALPHAS[2]),
    "strong":     WHITE,
}
SBC_BAND_68 = REGIONS_SEQ["marginal"]
SBC_BAND_95 = "#E3E3E3"
GMM_REF_COLOR = "#1B7837"
SBC_BAND_COLOR = "#808080"
SBC_BAND_ALPHA = (0.20, 0.09)
SBC_BANDS = {"68": "#E6E6E6", "95": "#F4F4F4"}
HATCH_FACE, HATCH_COLOR = "#F2F2F2", GUIDE

VOXEL_PDF = {
    "clean":        dict(color=INK, ls="-", lw=1.1),
    "mean removed": dict(color=_at(SEQ_NAME, 0.65), ls="-", lw=1.3),
    **{h: dict(color=NOISE_GREYS[h], ls=NOISE_LS[h], lw=1.2) for h in NOISE_HOURS},
}
CHROMA = "cmr.chroma"
COSMIC = "cmr.cosmic"
ABLATION = {
    "full":  _at(COSMIC, 0.45),
    "comp":  _at(CHROMA, 0.42),
    "small": _at(CHROMA, 0.60),
}
TOY_COLORS = {
    "cal": INK, "bias": _at(CHROMA, 0.42), "over": _at(CHROMA, 0.13),
    "under": _at(COSMIC, 0.85), "both": _at(CHROMA, 0.56), "theta": _at(CHROMA, 0.26),
    "skew": _at(COSMIC, 0.58),
}


def _ac():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import aa_color_style as ac
    return ac


def checks():
    """[(name, pass?, detail)]: every role in normal vision + the three dichromacies."""
    ac = _ac()
    views = ["normal", *ac.CVD_KINDS]

    def sim(c, v):
        rgb = np.array([mpl.colors.to_rgb(x) for x in c])
        return rgb if v == "normal" else ac.simulate_cvd(rgb, v)

    def min_pair(cols):
        return {v: min(ac._de(a, b) for i, a in enumerate(sim(cols, v))
                       for b in sim(cols, v)[i + 1:]) for v in views}

    def cross(c1, c2):
        return {v: min(ac._de(a, b) for a in sim(c1, v) for b in sim(c2, v)) for v in views}

    def fmt(d):
        return "  ".join(f"{k[:4]} {v:.0f}" for k, v in d.items())

    out = []
    arms = [ARMS[k][1] for k in ARMS]
    main = [ARMS[k][1] for k in MAIN_ARMS]
    for name, cols, thr in (("all 10 arms, every pair", arms, 9),
                            ("6 main arms, every pair", main, 15),
                            ("redshift colours, every pair", list(Z_COLORS.values()), 15),
                            ("voxel-PDF curves, every pair",
                             [d["color"] for d in VOXEL_PDF.values()], 10)):
        d = min_pair(cols)
        out.append((f"{name} (dE >= {thr})", min(d.values()) >= thr, fmt(d)))
    d = cross(list(Z_COLORS.values()), main)
    out.append(("redshift vs main-arm colours (dE >= 10)", min(d.values()) >= 10, fmt(d)))
    r = ac.cvd_check(SEQ_CMAP, "sequential")
    out.append(("SEQ_CMAP: lightness monotonic, range >= 40", r["cls"] == "friendly",
                r["why"].split(":", 1)[1]))
    cm = DIV_CMAP
    ends = {v: min(ac._de(sim([cm(x)], v)[0], sim([cm(1 - x)], v)[0])
                   for x in (0.0, 0.1, 0.2, 0.3)) for v in views}
    out.append(("DIV_CMAP: -dTb vs +dTb apart, |pos| >= 0.2 (dE >= 20)",
                min(ends.values()) >= 20, fmt(ends)))
    steps = list(REGIONS.values())
    d = {v: min(ac._de(a, b) for a, b in zip(sim(steps, v)[:-1], sim(steps, v)[1:]))
         for v in views}
    out.append(("REGIONS: neighbouring steps (dE >= 4, + contour lines)",
                min(d.values()) >= 4, fmt(d)))
    return out


def preview(name="paper_colors_preview"):
    """A4 sheet: every role in the four colour visions + the checks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ac = _ac()
    views = ["normal", *ac.CVD_KINDS]
    rows = [("ARMS (validation)", None)]
    for fam, (cm, hdr, mk) in ARM_FAMILY.items():
        keys = [k for k, (f, _, _) in ARM_POS.items() if f == fam]
        rows.append((f"  {hdr}  ({mk})", [ARMS[k][1] for k in keys]))
    rows += [("NOISE greys 100 / 50 / 25 h", list(NOISE_GREYS.values())),
             ("ORDERED: SEQ_CMAP " + SEQ_NAME, SEQ_CMAP),
             ("  redshift 12.06 / 10.32 / 8.18", list(Z_COLORS.values())),
             ("SIGNED: DIV_CMAP " + DIV_NAME, DIV_CMAP),
             ("REGIONS (chi_hat, inner -> outer)", list(REGIONS.values())),
             ("SBC bands 68 / 95 %, hatch", list(SBC_BANDS.values()) + [HATCH_FACE]),
             ("voxel PDF: clean, mean rem., 100/50/25 h",
              [d["color"] for d in VOXEL_PDF.values()]),
             ("neutrals INK / MUTED / GUIDE", [INK, MUTED, GUIDE])]
    W, H = 11.69, 8.27
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(0.3, 0.4, "Paper colour scheme (sty/paper_colors.py)", fontsize=13,
            weight="bold", va="top", color=INK)
    x0, sw, gap, rh, y = 3.4, 1.85, 0.12, 0.36, 1.25
    for j, v in enumerate(views):
        ax.text(x0 + j * (sw + gap) + sw / 2, y - 0.1, v, ha="center", fontsize=9,
                style="italic", color=INK)
    g = np.linspace(0, 1, 256)
    for nm, obj in rows:
        if obj is None:
            ax.text(0.3, y + rh / 2, nm, fontsize=9, weight="bold", va="center", color=INK)
            y += rh
            continue
        ax.text(0.3, y + rh / 2, nm, fontsize=8, va="center", color=INK)
        cont = isinstance(obj, mpl.colors.Colormap)
        rgb = obj(g)[:, :3] if cont else np.array([mpl.colors.to_rgb(c) for c in obj])
        for j, v in enumerate(views):
            img = rgb if v == "normal" else ac.simulate_cvd(rgb, v)
            x = x0 + j * (sw + gap)
            if cont:
                ax.imshow(img[None], aspect="auto", extent=(x, x + sw, y + rh * 0.85, y + 0.05),
                          interpolation="bilinear")
            else:
                for i, c in enumerate(img):
                    ax.add_patch(Rectangle((x + i * sw / len(img), y + 0.05), sw / len(img),
                                           rh * 0.8, fc=c, ec="none"))
            ax.add_patch(Rectangle((x, y + 0.05), sw, rh * 0.8, fc="none", ec=INK, lw=0.4))
        y += rh
    y += 0.25
    ax.text(0.3, y, "Checks (CIELAB dE76; normal, deuteranopia, protanopia, tritanopia)",
            fontsize=9, weight="bold", color=INK)
    y += 0.3
    res = checks()
    for nm, ok, det in res:
        ax.text(0.3, y, ("PASS  " if ok else "FAIL  ") + nm, fontsize=7.5,
                color="#1b6e3a" if ok else "#a33333", family="monospace")
        ax.text(6.4, y, det, fontsize=7, color=MUTED, family="monospace")
        y += 0.24
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=130)
    plt.close(fig)
    for nm, ok, det in res:
        print(("PASS " if ok else "FAIL ") + nm + "   " + det)
    print("wrote", out)


if __name__ == "__main__":
    preview()
