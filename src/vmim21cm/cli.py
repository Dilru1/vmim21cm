"""Command-line entry point: vmim21cm <command> CONFIG [options]."""
import argparse
import importlib

COMMANDS = {
    "stage1": ("vmim21cm.stages.stage1", "train the compressor and export t (Path A / B)"),
    "merge": ("vmim21cm.stages.merge", "join CNN export shards"),
    "stage2": ("vmim21cm.stages.stage2", "train the neural likelihood q(t | theta)"),
    "stage3": ("vmim21cm.stages.stage3", "MCMC posteriors for the SBC targets"),
    "stage4": ("vmim21cm.stages.stage4", "SBC calibration and posterior volume (CONFIG = eval file)"),
    "run": ("vmim21cm.stages.pipeline", "stages 1-3 for every seed pair of CONFIG on this machine"),
    "demo": ("vmim21cm.synthetic", "write tiny synthetic data + configs into directory CONFIG"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vmim21cm", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    mods = {}
    for name, (mod, help_) in COMMANDS.items():
        mods[name] = importlib.import_module(mod)
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("config")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. --set nle.n_layers=4 (repeatable)")
        mods[name].add_args(p)
    args = ap.parse_args(argv)
    mods[args.command].run(args)


if __name__ == "__main__":
    main()
