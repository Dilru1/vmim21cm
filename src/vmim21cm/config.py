"""YAML configs with ${VAR} expansion and key=value overrides, run directories, seeding, devices."""
import os
import random
from pathlib import Path

import numpy as np
import yaml


def _expand(x):
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    if isinstance(x, str):
        out = os.path.expandvars(x)
        if "$" in out:
            raise KeyError(f"unset environment variable in config value {x!r}")
        return out
    return x


def apply_overrides(cfg, overrides):
    """Set dotted keys from 'a.b.c=value' strings; values are parsed as YAML."""
    for item in overrides or ():
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path, overrides=None):
    """Read a YAML config, apply overrides, then expand ${VAR} from the environment."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _expand(apply_overrides(cfg, overrides))


def run_dir(cfg, cs):
    """{out_root}/{name}/c{cs}, or {out_root}/{name}/raw when there is no compressor."""
    root = Path(cfg["out_root"]) / cfg["name"]
    return root / "raw" if cfg["compressor"]["type"] == "none" else root / f"c{cs}"


def summaries_path(cfg, cs):
    """Summary table that stage 2 reads: the input table, or the stage-1 export."""
    if cfg["compressor"]["type"] == "none":
        return Path(cfg["data"]["summaries_npz"])
    return run_dir(cfg, cs) / "summaries"


def set_all_seeds(seed, deterministic=True):
    """Seed python, numpy and torch; cudnn deterministic or benchmark mode."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(name="auto"):
    """'auto' -> cuda, then mps, then cpu; anything else is passed to torch.device."""
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def n_cpus():
    """CPUs usable by this process (affinity-aware where the OS supports it)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
