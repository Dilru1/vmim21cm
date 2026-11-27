#!/usr/bin/env python3
"""Stage 2: train one or several NLE families q(t | theta) on the SAME exported summaries, chunked."""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs, split_by_sim
from sbi.torch_loader import make_loader
from sbi.nle import build_nle, nle_subdir, save_nle

from sbi.seeding import resolve_seeds, set_all_seeds, apply_arm_name


def run_pass(model, loader, opt, device, train, tag, log_every=1000, clip=5.0, t_noise=0.0):
    """t_noise > 0: fresh N(0, t_noise^2) added to every (standardized) t of every batch, train AND val -- ..."""
    model.train() if train else model.eval()
    total, n = 0.0, 0
    tic = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (th, t) in enumerate(loader):
            th = th.to(device, non_blocking=True).float()
            t = t.to(device, non_blocking=True).float()
            if t_noise > 0:
                t = t + t_noise * torch.randn_like(t)
            loss = -model.log_prob(th, t).mean()
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at batch {batch_idx}, skipping", flush=True)
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            total += loss.item(); n += 1
            if train and batch_idx % log_every == 0 and batch_idx > 0:
                dt = time.time() - tic; tic = time.time()
                eta = (len(loader) - batch_idx) / (log_every / max(dt, 1e-9))
                print(f"  batch {batch_idx}/{len(loader)} | "
                      f"loss {total/max(n,1):.4f} | eta {eta/60:.1f} min [{tag}]", flush=True)
    return total / max(n, 1)


def train_legacy(model, kind, subdir, nc_k, device, th_tr, t_tr, th_va, t_va,
                 t_mean, t_std, t_dim, n_params):
    """The TF baseline schedule (NDE_pdf_5param_multi_z.py), used when param_style == 'semelin_legacy':"""
    opt_name = str(nc_k.get("optimizer", "nadam")).lower()
    betas = tuple(float(b) for b in nc_k.get("betas", [0.5, 0.999]))
    if opt_name != "nadam":
        sys.exit(f"legacy training expects optimizer: nadam, got {opt_name!r}")
    print(f"[legacy] TF schedule: NAdam betas={betas} eps=1e-7, fresh optimizer per "
          f"phase, no clipping, final weights kept", flush=True)

    pin = device.type == "cuda"
    nsplit = int(nc_k["nsplit"])
    loss_log = []
    for pname, lr, nch in [("p1", float(nc_k["lr_phase1"]), nsplit),
                           ("p2", float(nc_k["lr_phase2"]), max(1, nsplit // 2))]:
        opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=betas, eps=1e-7)
        for j in range(nch):
            ts, te = j * len(th_tr) // nsplit, (j + 1) * len(th_tr) // nsplit
            vs, ve = j * len(th_va) // nsplit, (j + 1) * len(th_va) // nsplit
            tl = make_loader(th_tr[ts:te], t_tr[ts:te], nc_k["batch_size"], True,
                             nc_k.get("num_workers", 2), pin)
            vl = make_loader(th_va[vs:ve], t_va[vs:ve], nc_k["batch_size"], False,
                             nc_k.get("num_workers", 2), pin)
            tr_loss = run_pass(model, tl, opt, device, True,
                               f"{kind} {pname} chunk {j+1}/{nch}", clip=None)
            va_loss = run_pass(model, vl, opt, device, False,
                               f"{kind} {pname} val {j+1}/{nch}", clip=None)
            loss_log.append([len(loss_log) + 1, tr_loss, va_loss])
            np.save(os.path.join(subdir, "loss_history.npy"), np.array(loss_log))
            print(f"[{kind}|{pname}|legacy] chunk {j+1}/{nch} | train {tr_loss:.4f} | "
                  f"val {va_loss:.4f}", flush=True)

    save_nle(model, kind, subdir, nc_k, t_dim, n_params, t_mean, t_std)
    torch.save(model.state_dict(), os.path.join(subdir, "nle_model_final.pt"))
    print(f"[{kind}] done (legacy). final val NLL {loss_log[-1][2]:.4f} -> {subdir}", flush=True)
    return loss_log[-1][2]


def train_family(kind, nc, nle_root, device, th_tr, t_tr, th_va, t_va,
                 t_mean, t_std, t_dim, n_params, train_sims, val_sims,
                 init_seed=0, split_seed=42):
    """Train ONE family end-to-end (fresh model/opt/best-val), save to its subdir."""
    nc_k = copy.deepcopy(nc)
    nc_k["model"] = kind
    nc_k["init_seed"] = init_seed
    fam_offset = sum(ord(ch) for ch in kind) % 977
    set_all_seeds(init_seed * 1000 + fam_offset, verbose=False)
    kind, model = build_nle(nc_k, t_dim, n_params)
    model.to(device)
    subdir = nle_subdir(nle_root, nc_k)
    print(f"\n===== NLE family '{kind}' -> {subdir} =====", flush=True)
    np.save(os.path.join(subdir, "train_sims.npy"), train_sims)
    np.save(os.path.join(subdir, "val_sims.npy"), val_sims)
    t_noise = float(nc_k.get("t_noise_sigma", 0.0))
    with open(os.path.join(subdir, "run_meta.json"), "w") as f:
        json.dump({"family": kind, "init_seed": init_seed,
                   "split_seed": split_seed, "t_dim": t_dim,
                   "t_noise_sigma": t_noise}, f, indent=2)
    if t_noise > 0:
        if kind == "gmm" and nc_k.get("param_style") == "semelin_legacy":
            sys.exit("[t-noise] not implemented for the legacy GMM schedule")
        print(f"[t-noise] adding N(0, {t_noise}^2) to standardized t in every batch "
              f"(train + val); stage3 must use the same nle.t_noise_sigma", flush=True)


    if kind == "gmm" and nc_k.get("param_style") == "semelin_legacy":
        return train_legacy(model, kind, subdir, nc_k, device, th_tr, t_tr, th_va, t_va,
                            t_mean, t_std, t_dim, n_params)


    opt = torch.optim.Adam(model.parameters(), lr=float(nc_k["lr_phase1"]))

    pin = device.type == "cuda"
    best_val = float("inf")
    loss_log = []

    nsplit = int(nc_k["nsplit"])
    for pname, lr, nch in [("p1", float(nc_k["lr_phase1"]), nsplit),
                           ("p2", float(nc_k["lr_phase2"]), max(1, nsplit // 2))]:
        for g in opt.param_groups:
            g["lr"] = lr
        for j in range(nch):
            ts, te = j * len(th_tr) // nch, (j + 1) * len(th_tr) // nch
            vs, ve = j * len(th_va) // nch, (j + 1) * len(th_va) // nch
            tl = make_loader(th_tr[ts:te], t_tr[ts:te], nc_k["batch_size"], True,
                             nc_k.get("num_workers", 2), pin)
            vl = make_loader(th_va[vs:ve], t_va[vs:ve], nc_k["batch_size"], False,
                             nc_k.get("num_workers", 2), pin)
            tr_loss = run_pass(model, tl, opt, device, True, f"{kind} {pname} chunk {j+1}/{nch}",
                               t_noise=t_noise)
            va_loss = run_pass(model, vl, opt, device, False, f"{kind} {pname} val {j+1}/{nch}",
                               t_noise=t_noise)
            loss_log.append([len(loss_log) + 1, tr_loss, va_loss])
            np.save(os.path.join(subdir, "loss_history.npy"), np.array(loss_log))
            print(f"[{kind}|{pname}] chunk {j+1}/{nch} | train {tr_loss:.4f} | val {va_loss:.4f}",
                  flush=True)
            if va_loss < best_val:
                best_val = va_loss
                save_nle(model, kind, subdir, nc_k, t_dim, n_params, t_mean, t_std)
                print(f"  saved best ({best_val:.4f}) -> {subdir}", flush=True)

    torch.save(model.state_dict(), os.path.join(subdir, "nle_model_final.pt"))
    print(f"[{kind}] done. best val NLL {best_val:.4f}", flush=True)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--models", default=None,
                    help="comma-separated families to train sequentially in this job, "
                         "e.g. gmm,maf,nsf. Overrides yaml nle.models / nle.model.")
    ap.add_argument("--raw-t", dest="standardize", action="store_false", default=None,
                    help="train the NLE on RAW t (no per-dim standardization); "
                         "outputs go to {arm}/nle/raw_t/<model>/. Default (and "
                         "--standardize) uses standardized t -> {arm}/nle/standard_t/<model>/.")
    ap.add_argument("--standardize", dest="standardize", action="store_true", default=None,
                    help="force standardized t (this is the default).")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    nc = cfg["nle"]
    split_seed, init_seed = resolve_seeds(nc)
    apply_arm_name(cfg)
    dirs = arm_dirs(cfg)
    set_all_seeds(init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.standardize is not None:
        nc["standardize"] = bool(args.standardize)
    else:
        nc["standardize"] = bool(nc.get("standardize", True))

    if args.models:
        families = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    elif nc.get("models"):
        families = [str(m).lower() for m in nc["models"]]
    else:
        families = [nc.get("model", "gmm").lower()]

    sd = str(dirs["summaries"])
    theta = np.load(os.path.join(sd, "theta.npy"), mmap_mode="r")
    t = np.load(os.path.join(sd, "t.npy"), mmap_mode="r")
    sim_ids = np.load(os.path.join(sd, "original_sim_ids.npy"), mmap_mode="r")
    noise_ids = np.load(os.path.join(sd, "noise_ids.npy"), mmap_mode="r")
    cond_params = int(nc.get("cond_params", theta.shape[1]))
    if cond_params > theta.shape[1]:
        z = np.load(cfg["data"]["summaries_npz"])
        src_th, src_sim = z["theta"], z["original_sim_ids"]
        u, first = np.unique(src_sim, return_index=True)
        per_sim = src_th[first]
        sim_np = np.asarray(sim_ids)
        idx = np.searchsorted(u, sim_np)
        assert np.all(u[idx] == sim_np), "[cond5] sim id missing in summaries_npz"
        th_full = per_sim[idx].astype(np.float32)
        theta = np.asarray(theta, np.float32)
        assert np.allclose(th_full[:, :theta.shape[1]], theta, atol=1e-6), \
            "[cond5] exported theta != summaries_npz theta for the same sims"
        theta = np.concatenate([theta, th_full[:, theta.shape[1]:cond_params]], 1)
        del z, src_th, src_sim
        print(f"[cond5] NLE conditions on {cond_params} params (extra cols appended from "
              f"{cfg['data']['summaries_npz']}); subdir suffix '{nc.get('subdir_suffix', '')}'",
              flush=True)
    n_params, t_dim = theta.shape[1], t.shape[1]
    print(f"Stage2 arm={cfg['arm_name']} families={families} "
          f"rows={len(theta):,} t_dim={t_dim} n_params={n_params} device={device} "
          f"split_seed={split_seed} init_seed={init_seed} "
          f"seed_subdir={bool(nc.get('seed_subdir', False))}", flush=True)

    (th_tr, t_tr, sim_tr, _, th_va, t_va, sim_va, _, train_sims, val_sims) = split_by_sim(
        np.asarray(theta), np.asarray(t), np.asarray(sim_ids),
        np.asarray(noise_ids)[:, 0] if np.asarray(noise_ids).ndim == 2 else np.asarray(noise_ids),
        nc.get("val_frac", 0.1), split_seed)

    if nc["standardize"]:
        t_mean = t_tr.mean(0).astype(np.float32)
        t_std = (t_tr.std(0) + 1e-8).astype(np.float32)
        t_tr = ((t_tr - t_mean) / t_std).astype(np.float32)
        t_va = ((t_va - t_mean) / t_std).astype(np.float32)
        print(f"[t-scaling] STANDARDIZED: per-dim mean/std removed. "
              f"outputs -> {{arm}}/nle/standard_t/<model>/", flush=True)
    else:
        t_mean = np.zeros(t_dim, np.float32)
        t_std = np.ones(t_dim, np.float32)
        t_tr = t_tr.astype(np.float32)
        t_va = t_va.astype(np.float32)
        print(f"[t-scaling] RAW t (no standardization; identity mean/std saved). "
              f"outputs -> {{arm}}/nle/raw_t/<model>/", flush=True)
    th_tr = th_tr.astype(np.float32); th_va = th_va.astype(np.float32)

    results = {}
    for kind in families:
        results[kind] = train_family(kind, nc, dirs["nle"], device,
                                     th_tr, t_tr, th_va, t_va, t_mean, t_std,
                                     t_dim, n_params, train_sims, val_sims,
                                     init_seed=init_seed, split_seed=split_seed)

    scope = "standard_t" if nc["standardize"] else "raw_t"
    print(f"\n===== summary [{scope}] (best val NLL) =====", flush=True)
    for kind, v in results.items():
        print(f"  {kind:6s}  {v:.4f}", flush=True)
    if not nc["standardize"]:
        print("  NOTE: raw-t NLL is in a DIFFERENT coordinate system than "
              "standard-t NLL -- the two are NOT directly comparable as numbers. "
              "Compare via SBC/GV in stage 4, not by this loss.", flush=True)


if __name__ == "__main__":
    main()
