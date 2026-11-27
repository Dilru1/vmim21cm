#!/usr/bin/env python3
"""arm_look.py -- ONE look per configuration (arm) for fig 6 (corner), fig 7 (b, s) and fig B1 (SBC ..."""
from pathlib import Path

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import aa_style as aa
import paper_colors as pc

ROOT = Path(__file__).resolve().parents[3]

import os as _os
if _os.environ.get("AA_LOOK") == "classic":
    FAMILY_MAP = {"A": pc.ARM_FAMILY["A"][0], "V": pc.ARM_FAMILY["VMIM"][0],
                  "M": pc.ARM_FAMILY["MSE"][0]}
    NOISE_POS = {"A": {100: 0.45}, "V": {100: 0.25, 50: 0.50, 25: 0.75},
                 "M": {100: 0.45, 50: 0.65, 25: 0.85}}
else:
    FAMILY_MAP = {f: "cmr.chroma" for f in ("A", "V", "M")}
    NOISE_POS = {"A": {100: 0.70},
                 "V": {100: 0.13, 50: 0.24, 25: 0.33},
                 "M": {100: 0.42, 50: 0.52, 25: 0.60}}
K_MARKER = {4: "o", 8: "s", 16: "^"}
K_LS = {8: "-", 4: aa.DASH, 16: (0, (1, 1.2))}
REF_MARKER = {"ref": "*", "gmm": "X"}
REF_COLOR = {"ref": "black", "gmm": "black"}
GMM_COLOR = pc.GMM_REF_COLOR
GMM_LW_SCALE = 1.9


def line_width(key, lw):
    return lw * GMM_LW_SCALE if key == "gmm" else lw


def line_z(key, z):
    return 2.0 if key == "gmm" else z
NSF_COLOR = "black"


def parse(key):
    """(family, k, hours) of a key; family in A, V, M, ref, gmm."""
    if key in ("ref", "gmm"):
        return key, None, None
    if key.startswith("A_"):
        return "A", 4 if key == "A_mse4" else int(key[3:]), 100
    k, h = (int(x) for x in key[1:].split("_"))
    return key[0], k, h


FAMILY_K = {}
if _os.environ.get("AA_LOOK") != "classic":
    FAMILY_K[("V", 8)] = ("cmr.cosmic", {100: 0.45, 50: 0.58, 25: 0.70})
    FAMILY_K[("A", 8)] = ("cmr.cosmic", {100: 0.85})
    FAMILY_K[("A", 16)] = ("cmr.cosmic", {100: 0.95})


def colour(key):
    fam, k, h = parse(key)
    if fam in ("ref", "gmm"):
        return REF_COLOR[fam]
    if (fam, k) in FAMILY_K:
        cmap, pos = FAMILY_K[(fam, k)]
        return pc._at(cmap, pos[h])
    return pc._at(FAMILY_MAP[fam], NOISE_POS[fam][h])


def marker(key):
    fam, k, _ = parse(key)
    return REF_MARKER[fam] if fam in ("ref", "gmm") else K_MARKER[k]


def filled(key):
    return key != "A_mse4"


def linestyle(key):
    fam, k, _ = parse(key)
    return "-" if fam in ("ref", "gmm") else K_LS[k]


def line_colour(key):
    return {"ref": NSF_COLOR, "gmm": GMM_COLOR}.get(key) or colour(key)


def label(key):
    fam, k, h = parse(key)
    if fam == "ref":
        return "PDF+PS, NSF"
    if fam == "gmm":
        return "PDF+PS, GMM"
    return rf"$k={k}$, {h} h"


def header(key):
    fam = parse(key)[0]
    if fam == "A":
        return "Path A (MLP), MSE" if key == "A_mse4" else "Path A (MLP), VMIM"
    return {"ref": "Reference", "gmm": "Reference",
            "V": "Path B (CNN), VMIM", "M": "Path B (CNN), MSE"}[fam]


_A = ROOT / "paper_out" / "nov21" / "patha" / "global"
_B = ROOT / "paper_out" / "nov21" / "pathb" / "global"
DROP_100H = " c45 "
SOURCES = {
    "gmm":    (ROOT / "paper_out" / "nov19" / "reference", "ref_pdfps_gmm"),
    "ref":    (ROOT / "paper_out" / "nsf_depth" / "baseline_L8h128_vs_L8_L4",
               "pdf_ps_nsf_L8h64"),
    "A_k16":  (_A, "patha_vmim_k16_L8h64"),
    "A_k8":   (_A, "patha_vmim_k8_L8h64"),
    "A_k4":   (_A, "patha_vmim_k4_L8h64"),
    "A_mse4": (_A, "patha_mse_k4_L8h64"),
    **{f"V8_{h}": (_B, f"pathb_vmim_k8_{n}_L8h128", *([DROP_100H] if h == 100 else []))
       for h, n in ((100, "n1"), (50, "nsqrt2"), (25, "n2"))},
    **{f"V4_{h}": (_B, f"pathb_vmim_k4_{n}_L8h128", *([DROP_100H] if h == 100 else []))
       for h, n in ((100, "n1"), (50, "nsqrt2"), (25, "n2"))},
    **{f"M4_{h}": (_B, f"pathb_mse_k4_{n}_L8h128", *([DROP_100H] if h == 100 else []))
       for h, n in ((100, "n1"), (50, "nsqrt2"), (25, "n2"))},
}
MODE = "filtered_dlogp10"
LINE_KEYS = ["gmm", "ref", "A_k8", "V8_100", "V8_50", "V4_100", "V4_50", "M4_100", "M4_50"]


def runs(key):
    """Run dicts (name, dir, n_params, rank_hist, used, ...) of one arm from metrics.json."""
    import json
    d, grp, *drop = SOURCES[key]
    rep = json.load(open(Path(d) / "metrics.json"))[MODE]
    sel = [a for a in rep if a.get("group") == grp
           and not any(x in f" {a['name']} " for x in drop)]
    if not sel:
        raise SystemExit(f"no runs for {key}: {d} group {grp}")
    return sel


LEG_FS = aa.FS["note"] + 0.2
ROW_H = 3.1
HANDLE_W = 5.0


def _handle(fig, W, H, x, y, key, lw):
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / H)
    fig.add_artist(Line2D([fx(x), fx(x + HANDLE_W)], [fy(y)] * 2, color=line_colour(key),
                          ls=linestyle(key), lw=line_width(key, lw), transform=fig.transFigure))
    mk = marker(key)
    ms = {"o": 3.0, "s": 2.8, "^": 3.4, "*": 5.4, "X": 4.0}[mk]
    kw = (dict(mfc=colour(key), mec=pc.INK, mew=0.4) if key not in ("ref", "gmm")
          else dict(mfc=line_colour(key), mec="white", mew=0.4))
    fig.add_artist(Line2D([fx(x + HANDLE_W / 2)], [fy(y)], ls="none", marker=mk, ms=ms,
                          transform=fig.transFigure, **kw))


def groups(keys):
    """[(header, [keys])] in key order."""
    out = {}
    for k in keys:
        out.setdefault(header(k), []).append(k)
    return list(out.items())


def default_columns(keys, extra=()):
    """Legend columns for the line plots: [Reference + Path A] [Path B VMIM] [Path B MSE] [extra]."""
    g = dict(groups(keys))
    cols = [[(h, g[h]) for h in ("Reference", "Path A (MLP), VMIM", "Path A (MLP), MSE")
             if h in g]]
    cols += [[(h, g[h])] for h in ("Path B (CNN), VMIM", "Path B (CNN), MSE") if h in g]
    cols += [[x] for x in extra]
    return [c for c in cols if c]


SHORT_HEADER = {"Path A (MLP), VMIM": "Path A, VMIM", "Path A (MLP), MSE": "Path A, MSE",
                "Path B (CNN), VMIM": "Path B, VMIM",
                "Path B (CNN), MSE": "Path B, MSE"}


def legend_height(columns):
    return max(sum(1 + len(it) for _, it in col) for col in columns) * ROW_H


def legend_below(fig, W, H, x0, x1, y_top, columns, lw=0.8, col_w=None, short=True):
    """Draw the legend columns in the strip [x0, x1] mm from y_top mm (from the top); returns the height ..."""
    fx, fy = (lambda v: v / W), (lambda v: 1 - v / H)
    rel = col_w or [1.0] * len(columns)
    xs, x = [], x0
    for r in rel:
        xs.append(x)
        x += (x1 - x0) * r / sum(rel)
    for col, xc in zip(columns, xs):
        y = y_top - 0.5 * ROW_H
        for hdr, items in col:
            y += ROW_H
            if hdr:
                fig.text(fx(xc), fy(y), SHORT_HEADER.get(hdr, hdr) if short else hdr,
                         ha="left", va="center_baseline", fontsize=LEG_FS, style="italic",
                         color=pc.INK)
            for it in items:
                y += ROW_H
                if isinstance(it, str):
                    _handle(fig, W, H, xc, y, it, lw)
                    txt = label(it)
                else:
                    draw, txt = it
                    draw(fig, W, H, xc, y)
                fig.text(fx(xc + HANDLE_W + 1.0), fy(y), txt, ha="left",
                         va="center_baseline", fontsize=LEG_FS, color=pc.INK)
    return legend_height(columns)
