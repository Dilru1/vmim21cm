#!/usr/bin/env python3
"""Stage 1: train the arm's compressor and export NLE summaries."""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs, load_source, split_by_sim
from sbi.compressors.heads import build_head_from_cfg
from sbi.compressors.objectives import (build_objective, objective_name,
                                        needs_head as obj_needs_head)
from sbi.train_comp_update import run_training
from sbi.seeding import (resolve_seeds, set_all_seeds, seed_worker,
                         make_generator, apply_arm_name)


import torch

class NoiseResampledDataset(torch.utils.data.Dataset):
    """One row per SIM; a fresh random noise rep is drawn each __getitem__."""
    def __init__(self, t_all, th_all, sim_ids_all, sims, n_target=4):
        self.n_target = int(n_target)
        self.groups = []
        for s in sims:
            m = sim_ids_all == s
            self.groups.append((th_all[m][0], t_all[m]))
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, i):
        th, block = self.groups[i]
        j = np.random.randint(block.shape[0])
        return torch.from_numpy(block[j].astype(np.float32)), torch.from_numpy(th[:self.n_target].astype(np.float32))


def export_memmap(compressor, get_batch_iter, n_rows, device, out_dir, t_dim,
                  noise_scale=None, summary=None):
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    raw.eval()
    theta_mm = np.lib.format.open_memmap(os.path.join(out_dir, "theta.npy"), mode="w+", dtype=np.float32, shape=(n_rows, 4))
    t_mm = np.lib.format.open_memmap(os.path.join(out_dir, "t.npy"), mode="w+", dtype=np.float32, shape=(n_rows, t_dim))
    sim_mm = np.lib.format.open_memmap(os.path.join(out_dir, "original_sim_ids.npy"), mode="w+", dtype=np.int64, shape=(n_rows,))
    noise_mm = np.lib.format.open_memmap(os.path.join(out_dir, "noise_ids.npy"), mode="w+", dtype=np.int64, shape=(n_rows, 3))
    if noise_scale is not None:
        np.save(os.path.join(out_dir, "export_noise_scale.npy"), np.array([noise_scale], np.float32))
    off = 0
    with torch.no_grad():
        for bi, (x, theta, sim, nids) in enumerate(get_batch_iter()):
            x = x.to(device)
            out = raw(x); t = out[0] if isinstance(out, tuple) else out
            if summary is not None:
                t = summary(t)
            k = x.shape[0]
            theta_mm[off:off+k] = np.asarray(theta)[:, :4].astype(np.float32)
            t_mm[off:off+k] = t.detach().float().cpu().numpy().astype(np.float32)
            sim_mm[off:off+k] = np.asarray(sim).astype(np.int64)
            nn_ = np.asarray(nids)
            if nn_.ndim == 2 and nn_.shape[1] < 3:
                nn_ = np.concatenate([nn_, -np.ones((nn_.shape[0], 3 - nn_.shape[1]), nn_.dtype)], 1)
            noise_mm[off:off+k] = nn_.astype(np.int64) if nn_.ndim == 2 else np.repeat(nn_[:, None], 3, 1).astype(np.int64)
            off += k
            if bi % 100 == 0:
                print(f"  export {off:,}/{n_rows:,}", flush=True)
    for mm in (theta_mm, t_mm, sim_mm, noise_mm):
        mm.flush()
    print(f"exported summaries -> {out_dir}  t {t_mm.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--no-export", action="store_true",
                    help="train + RF-R2 probe + loss curves only; skip the memmap export "
                         "(for quick VMIM diagnostics where you just want the plots)")
    ap.add_argument("--ckpt", default="bestprobe", choices=["bestprobe", "bestloss"],
                    help="which checkpoint to export from: bestprobe (best mean RF-R2, "
                         "default -- the one where rHS survived) or bestloss")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--noise-start", type=int, default=0,
                    help="first noise realisation index for this shard")
    ap.add_argument("--noise-end", type=int, default=None,
                    help="one past the last noise index; None = all. When set, "
                         "the shard writes to summaries/part_<start>_<end>/ and "
                         "merge_export_shards.py concatenates them afterwards.")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.noise_end is not None and cfg.get("arm_type", "cnn") != "cnn":
        sys.exit("--noise-start/--noise-end are CNN-only: the mlp and raw export "
                 "branches ignore them and write straight to summaries/, so N "
                 "array tasks would write the same four memmaps at once.")
    c = cfg["compressor"]
    split_seed, init_seed = resolve_seeds(c)
    n_target = int(c.get("target_params", cfg["n_params"]))
    print(f"[target5] stage-1 target params: {n_target}  (inferred n_params: {cfg['n_params']})", flush=True)
    if n_target != cfg["n_params"] and cfg["arm_type"] != "mlp":
        sys.exit("[target5] compressor.target_params != n_params is only wired for arm_type=mlp")

    apply_arm_name(cfg)
    dirs = arm_dirs(cfg)

    set_all_seeds(init_seed, deterministic=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage1 arm={cfg['arm_name']} split_seed={split_seed} init_seed={init_seed}", flush=True)

    if cfg["arm_type"] == "mlp":
        from sbi.compressors.mlp import SummaryCompressor
        from sbi.torch_loader import make_loader
        src = load_source(cfg["data"]["summaries_npz"])
        n_input = src.n_summaries
        (t_tr, th_tr, sim_tr, ntr, t_va, th_va, sim_va, nva, train_sims, val_sims) = split_by_sim(
            src.t, src.theta, src.sim_ids, src.noise_ids, c["val_frac"], split_seed)
        np.save(dirs["nle"] / "train_sims.npy", train_sims)
        np.save(dirs["nle"] / "val_sims.npy", val_sims)

        compressor = SummaryCompressor(n_input=n_input, t_dim=cfg["t_dim"],
                                       dense_layers=tuple(c["dense_layers"]), dropout=c["dropout"],
                                       activation=c["activation"], use_resnet=c["use_resnet"],
                                       n_params=n_target,
                                       direct=bool(c.get("direct_regression", False))).to(device)

        from torch.utils.data import DataLoader

        train_ds = NoiseResampledDataset(t_tr, th_tr, sim_tr, train_sims, n_target)
        val_ds = NoiseResampledDataset(t_va, th_va, sim_va, val_sims, n_target)

        tl = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,
                        num_workers=c["num_workers"], pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker, generator=make_generator(init_seed))
        vl = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False,
                        num_workers=c["num_workers"], pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker)


        class _Wrap:
            def __init__(self, loader): self.loader = loader
            def __iter__(self):
                for a, b in self.loader:
                    yield a, b, torch.zeros(len(a)), torch.zeros(len(a), 3)
            def __len__(self): return len(self.loader)

        tl, vl = _Wrap(tl), _Wrap(vl)
        export_src = src
    else:

        model_choice = c.get("model", "seblock").lower()

        from sbi.cubes import prepare_cube_loaders, DeterministicNoisyExport
        if model_choice == "seblock":
            from sbi.compressors.cnn_seblock_up import ResNet3DCompressor as comp


        elif model_choice == "conv4d":
            from sbi.compressors.cnn_grn4d_up import Conv4DCompressor as comp

            from sbi.cubes_grn4d import prepare_cube_loaders, DeterministicNoisyExport


        from torch.utils.data import DataLoader

        tl, vl, full_va = prepare_cube_loaders(cfg["data"], c, split_seed, c["val_frac"])
        if model_choice == "seblock":
            from sbi.compressors.cnn_seblock_up import build_cnn_from_cfg
            compressor = build_cnn_from_cfg(c, t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                            direct=bool(c.get("direct_regression", False))).to(device)
        elif model_choice == "seblock_amp":
            from sbi.compressors.cnn_amp import build_amp_cnn_from_cfg
            compressor = build_amp_cnn_from_cfg(c, t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                                direct=bool(c.get("direct_regression", False))).to(device)
        elif model_choice == "cnn_ablate":
            from sbi.compressors.cnn_ablate import build_ablate_cnn_from_cfg
            compressor = build_ablate_cnn_from_cfg(c, t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                                   direct=bool(c.get("direct_regression", False))).to(device)
        elif model_choice == "seblock_ablate":
            from sbi.compressors.cnn_ablate import build_seblock_ablate_from_cfg
            compressor = build_seblock_ablate_from_cfg(c, t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                                       direct=bool(c.get("direct_regression", False))).to(device)
        elif model_choice == "simple":
            from sbi.compressors.cnn_model import build_simple_cnn_from_cfg
            compressor = build_simple_cnn_from_cfg(c, t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                                   direct=bool(c.get("direct_regression", False))).to(device)
        else:
            compressor = comp(t_dim=cfg["t_dim"], n_params=cfg["n_params"],
                                             direct=bool(c.get("direct_regression", False))).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            compressor = nn.DataParallel(compressor)

    import json
    with open(dirs["nle"] / "run_meta.json", "w") as f:
        json.dump({"arm_name": cfg["arm_name"], "arm_type": cfg["arm_type"],
                   "split_seed": split_seed, "init_seed": init_seed,
                   "t_dim": cfg["t_dim"], "model": c.get("model", "seblock"),
                   "objective": str(c.get("objective", "vmim")).lower(),
                   "aux_only": bool(c.get("aux_only", False)),
                   "target_params": n_target}, f, indent=2)

    obj_name = objective_name(c)
    head = None
    if obj_needs_head(c):
        head_cfg = dict(cfg)
        head_cfg["n_params"] = n_target
        if obj_name == "vib":
            head_cfg["t_dim"] = int(cfg["t_dim"]) // 2
        head = build_head_from_cfg(c, head_cfg).to(device)
        print("sigma floor for each parm:", getattr(head, "sigma_floor", None),
              "jitter for each parm", head.pre.jitter, flush=True)

    objective = build_objective(c, cfg, head=head,
                                t_dim=cfg["t_dim"],
                                n_params=n_target).to(device)
    export_t_dim = objective.out_dim(int(cfg["t_dim"]))
    print(f"[objective] {obj_name}  head={'yes' if head is not None else 'no'}  "
          f"t_dim {cfg['t_dim']} -> exported {export_t_dim}", flush=True)

    if not args.export_only:
        t0 = time.perf_counter()
        run_training(compressor, head, tl, vl, device, str(dirs["nle"]), c,
                     n_target, objective=objective)
        print(f"[TIMER] train {time.perf_counter()-t0:.1f}s", flush=True)

    if args.no_export:
        print("[NO-EXPORT] training + probes done; skipping memmap export. "
              f"RF-R2 history and loss curves saved in {dirs['nle']}", flush=True)
        return

    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    ckpt_name = "learned_compressor.pt" if args.ckpt == "bestprobe" else "learned_compressor.pt"
    ckpt_path = dirs["nle"] / ckpt_name
    if c.get("ckpt_from_arm"):
        import copy
        src_cfg = copy.deepcopy(cfg)
        src_cfg["arm_name"] = c["ckpt_from_arm"]
        apply_arm_name(src_cfg)
        ckpt_path = arm_dirs(src_cfg)["nle"] / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt_from_arm: no checkpoint at {ckpt_path}")
    if not ckpt_path.exists():
        ckpt_path = dirs["nle"] / "learned_compressor.pt"
        print(f"[WARN] requested {ckpt_name} missing; falling back to {ckpt_path.name}", flush=True)
    print(f"exporting from checkpoint: {ckpt_path}", flush=True)
    raw.load_state_dict(torch.load(ckpt_path, map_location=device)); raw.eval()
    tta = int(c.get("tta_group", 1))
    if tta > 1:
        from sbi.compressors.tta import SymmetrizedCompressor
        compressor = SymmetrizedCompressor(raw, n_group=tta).eval()
        print(f"[export] symmetrised over {tta} group elements", flush=True)

    if cfg["arm_type"] == "mlp":
        from sbi.torch_loader import make_loader
        from torch.utils.data import DataLoader, TensorDataset
        all_t = torch.tensor(export_src.t, dtype=torch.float32)
        all_theta = export_src.theta; all_sim = export_src.sim_ids; all_noise = export_src.noise_ids
        loader = DataLoader(TensorDataset(all_t, torch.arange(len(all_t))),
                            batch_size=c["save_batch_size"], shuffle=False)
        def it():
            for xb, idx in loader:
                idx = idx.numpy()
                yield xb, all_theta[idx], all_sim[idx], all_noise[idx] if all_noise.ndim == 2 else all_noise[idx]
        export_memmap(compressor, it, len(all_t), device, str(dirs["summaries"]),
                      export_t_dim, summary=objective.summary)
    else:
        from torch.utils.data import DataLoader
        total = cfg["data"]["total_nnoise"]
        n_start = args.noise_start
        n_end   = args.noise_end if args.noise_end is not None else total
        ds = DeterministicNoisyExport(full_va, total, c["noise_scale"],
                                      noise_start=n_start, noise_end=n_end)

        out_dir = dirs["summaries"]
        if args.noise_end is not None:
            out_dir = out_dir / f"part_{n_start:05d}_{n_end:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[export] shard noise[{n_start},{n_end}) -> {len(ds):,} rows -> {out_dir}",
              flush=True)

        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 6)); nw = min(8, max(0, cpus - 1))
        loader = DataLoader(ds, batch_size=c["save_batch_size"], shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=(nw > 0), prefetch_factor=4)
        def it():
            for x, theta, sim, nids in loader:
                yield x, theta, sim, nids
        export_memmap(compressor, it, len(ds), device, str(out_dir), export_t_dim,
                      noise_scale=c["noise_scale"], summary=objective.summary)


if __name__ == "__main__":
    main()
