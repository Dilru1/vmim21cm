"""Run stages 1-3 (and optionally 4) for every seed pair of one config on this machine."""
import os
import subprocess
import sys
import tempfile

import yaml

from ..config import load_config

EVAL_DEFAULTS = dict(thin=11, dlogp=10.0, burnin_frac=0.0, nbins=30, walker_cut=5.0)


def _ints(s):
    return [int(x) for x in s.split(",")] if s else None


def add_args(ap):
    ap.add_argument("--cseeds", default=None, help="comma list, default: seeds.compressor of the config")
    ap.add_argument("--nseeds", default=None, help="comma list, default: seeds.nle of the config")
    ap.add_argument("--stages", default="1,2,3", help="subset of 1,2,3 to run")
    ap.add_argument("--device", default="auto", help="stage-1 device: auto, cpu, cuda, mps")
    ap.add_argument("--threads", type=int, default=None, help="stage-3 torch threads")
    ap.add_argument("--eval", action="store_true", help="run stage 4 on this config's runs afterwards")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")


def run(args):
    cfg = load_config(args.config, args.set)
    none = cfg["compressor"]["type"] == "none"
    cseeds = [0] if none else (_ints(args.cseeds) or cfg["seeds"]["compressor"])
    nseeds = _ints(args.nseeds) or cfg["seeds"]["nle"]
    stages = set(_ints(args.stages))
    extra = [x for s in (args.set or []) for x in ("--set", s)]

    def call(*argv):
        cmd = [sys.executable, "-m", "vmim21cm", *map(str, argv), *extra]
        print("+", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    for cs in cseeds:
        if 1 in stages and not none:
            call("stage1", args.config, "--cseed", cs, "--device", args.device)
        for ns in nseeds:
            if 2 in stages:
                call("stage2", args.config, "--cseed", cs, "--nseed", ns)
            if 3 in stages:
                th = ["--threads", args.threads] if args.threads else []
                call("stage3", args.config, "--cseed", cs, "--nseed", ns, *th)

    if args.eval:
        e = dict(EVAL_DEFAULTS, **cfg.get("eval", {}))
        e.setdefault("params", list(range(int(cfg["n_params"]))))
        e.setdefault("param_names", [f"p{j}" for j in e["params"]])
        e["out"] = os.path.join(cfg["out_root"], "eval", cfg["name"])
        e["groups"] = [{"label": cfg["name"], "config": os.path.abspath(args.config),
                        "cseeds": cseeds, "nseeds": nseeds}]
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(e, f)
        call("stage4", f.name)
