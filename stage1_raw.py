#!/usr/bin/env python3
"""Stage 1 for the BASELINE arm: NO compression."""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs, load_source, split_by_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    c, dirs = cfg["compressor"], arm_dirs(cfg)

    src = load_source(cfg["data"]["summaries_npz"])
    print(f"Baseline arm={cfg['arm_name']} raw summaries: theta{src.theta.shape} t{src.t.shape}", flush=True)

    (_, _, _, _, _, _, _, _, train_sims, val_sims) = split_by_sim(
        src.t, src.theta, src.sim_ids, src.noise_ids, c["val_frac"], c["seed"])
    np.save(dirs["nle"] / "train_sims.npy", train_sims)
    np.save(dirs["nle"] / "val_sims.npy", val_sims)

    n_rows, n_t = src.t.shape[0], src.t.shape[1]
    out = dirs["summaries"]
    theta_mm = np.lib.format.open_memmap(out / "theta.npy", mode="w+", dtype=np.float32, shape=(n_rows, src.theta.shape[1]))
    t_mm = np.lib.format.open_memmap(out / "t.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_t))
    sim_mm = np.lib.format.open_memmap(out / "original_sim_ids.npy", mode="w+", dtype=np.int64, shape=(n_rows,))
    noise = src.noise_ids
    noise2d = noise if noise.ndim == 2 else np.repeat(np.asarray(noise)[:, None], 3, 1)
    noise_mm = np.lib.format.open_memmap(out / "noise_ids.npy", mode="w+", dtype=np.int64, shape=(n_rows, 3))

    theta_mm[:] = src.theta.astype(np.float32)
    t_mm[:] = src.t.astype(np.float32)
    sim_mm[:] = np.asarray(src.sim_ids).astype(np.int64)
    noise_mm[:] = noise2d.astype(np.int64)
    for mm in (theta_mm, t_mm, sim_mm, noise_mm):
        mm.flush()
    print(f"exported RAW summaries -> {out}  t {t_mm.shape} (no compression)", flush=True)


if __name__ == "__main__":
    main()
