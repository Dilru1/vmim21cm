"""COLOURS AND MARKERS OF THE VALIDATION FIGURES -- the one place to change them."""
import cmasher as cmr
try:
    import cmastro  # noqa: F401             # "cma:<name>"; fails on matplotlib >= 3.9
except Exception:
    cmastro = None

REF_COLOR = "#1A1A1A"
FAMILY_CMAPS = {
    "A":    ("cmr.emerald", (0.25, 0.70)),
    "VMIM": ("cmr.ocean",   (0.20, 0.75)),
    "MSE":  ("cmr.ember",   (0.30, 0.80)),
}
FAMILY_MARKERS = {"ref": "o", "A": "s", "VMIM": "D", "MSE": "^"}
FAMILY_MEMBERS = {
    "A":    [("A_k4", r"Path A, $k=4$"), ("A_k8", r"Path A, $k=8$"),
             ("A_k16", r"Path A, $k=16$")],
    "VMIM": [("VMIM_100", "VMIM 100 h"), ("VMIM_50", "VMIM 50 h"), ("VMIM_25", "VMIM 25 h")],
    "MSE":  [("MSE_100", "MSE 100 h"), ("MSE_50", "MSE 50 h"), ("MSE_25", "MSE 25 h")],
}

TOY_CASES = {
    "cal":   ("calibrated", REF_COLOR, "o"),
    "bias":  (r"bias $0.15\sigma$", ("cmr.ember", 0.55), "o"),
    "over":  ("12% too narrow", ("cmr.ocean", 0.45), "s"),
    "under": ("15% too wide", ("cmr.emerald", 0.50), "s"),
    "both":  (r"bias $0.12\sigma$ + 8% too narrow", ("cmr.amber", 0.60), "D"),
    "theta": (r"varying bias (rms $0.3\sigma$)", "#737373", "^"),
    "skew":  ("skewed (true mean and width)", ("cmr.lavender", 0.45), "v"),
}


def _hex(cmap, lo, hi, n):
    return cmr.take_cmap_colors(cmap, n, cmap_range=(lo, hi), return_fmt="hex")


ARMS = {"ref": ("PDF+PS (reference)", REF_COLOR, FAMILY_MARKERS["ref"]),
        "ref_gmm": ("PDF+PS, GMM", "#8C8C8C", "v")}
for _fam, _members in FAMILY_MEMBERS.items():
    _cm, (_lo, _hi) = FAMILY_CMAPS[_fam]
    for (_key, _lab), _c in zip(_members, _hex(_cm, _lo, _hi, len(_members))):
        ARMS[_key] = (_lab, _c, FAMILY_MARKERS[_fam])

def set_colors(colors, markers=None):
    """Override arms from a plotting script, BEFORE aa_style / _settings / _sbc / _corner / ..."""
    for k, c in colors.items():
        lab, _, mk = ARMS[k]
        if not isinstance(c, str):
            c = _hex(c[0], c[1], c[1], 1)[0]
        ARMS[k] = (lab, c, (markers or {}).get(k, mk))
    for k, mk in (markers or {}).items():
        if k not in colors:
            ARMS[k] = ARMS[k][:2] + (mk,)


TOY_LOOK = {k: (lab, c if isinstance(c, str) else _hex(c[0], c[1], c[1], 1)[0], mk)
            for k, (lab, c, mk) in TOY_CASES.items()}


OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
)

LINESTYLES = ("-", "--", "-.", ":")

FIGFMT = "pdf"


def arm_color(i):
    """Colour for arm index i, cycling past 8 arms."""
    return OKABE_ITO[i % len(OKABE_ITO)]


def arm_linestyle(i):
    """Line style for arm index i; changes only after a full colour cycle."""
    return LINESTYLES[(i // len(OKABE_ITO)) % len(LINESTYLES)]
