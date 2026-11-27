"""Seed control for ensemble / replica runs."""
import os
import random

import numpy as np


def resolve_seeds(block, legacy_key="seed"):
    """Return (split_seed, init_seed) from a config block."""
    legacy = block.get(legacy_key, 42)
    split_seed = int(block.get("split_seed", legacy))
    init_seed = int(block.get("init_seed", legacy))
    return split_seed, init_seed


def resolve_arm_name(cfg):
    """The replica-aware arm name."""
    c = cfg.get("compressor", {})
    name = cfg["arm_name"]
    if c.get("tag_arm_with_seed", False):
        _, comp_init = resolve_seeds(c)
        name = f"{name}_s{comp_init}"
    return name


def apply_arm_name(cfg):
    """Rewrite cfg['arm_name'] in place."""
    cfg["arm_name"] = resolve_arm_name(cfg)
    return cfg


def set_all_seeds(seed, deterministic=True, verbose=True):
    """Seed python / numpy / torch (CPU + all CUDA devices)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        if verbose:
            print(f"[seed] torch unavailable; seeded python+numpy with {seed}", flush=True)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    if verbose:
        print(f"[seed] global seed={seed} deterministic={deterministic}", flush=True)


def make_generator(seed):
    """torch.Generator for DataLoader(shuffle=True) so batch ORDER follows init_seed."""
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id):
    """DataLoader worker_init_fn: give every worker its own numpy/random stream."""
    import torch
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))
    random.seed(base + worker_id)


def seeded_loader_kwargs(init_seed, shuffle):
    """Convenience: the two kwargs every training DataLoader should carry."""
    kw = {"worker_init_fn": seed_worker}
    if shuffle:
        kw["generator"] = make_generator(init_seed)
    return kw
