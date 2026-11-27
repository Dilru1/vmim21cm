#!/usr/bin/env python3
"""Stage 3: SBC MCMC chains for the arm, per NLE family."""
import argparse
import json
import os
import sys
from pathlib import Path
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi import load_config, arm_dirs, load_source, load_sbc_targets
from sbi.seeding import apply_arm_name
from sbi.mcmc import run_chain

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.set_num_interop_threads(1)


def load_model_new(nle_root, nc, device):
    """New per-family layout: {arm}/nle/<scope>/<model>/ with model_config.json, where scope is ..."""
    import json
    from sbi.nle import load_nle
    kind = nc.get("model", "gmm").lower() + str(nc.get("subdir_suffix", ""))
    scope = "standard_t" if bool(nc.get("standardize", True)) else "raw_t"
    base = Path(nle_root) / scope / kind
    subdir = base
    if nc.get("seed_subdir", False):
        iseed = int(nc.get("init_seed", nc.get("seed", 42)))
        subdir = base / f"seed_{iseed}"
        if not (subdir / "model_config.json").exists():
            sys.exit(f"nle.seed_subdir=true but no model in {subdir}")
    elif not (subdir / "model_config.json").exists():
        seed_dirs = sorted(base.glob("seed_*")) if base.is_dir() else []
        hits = [d for d in seed_dirs if (d / "model_config.json").exists()]
        if len(hits) == 1:
            subdir = hits[0]
            print(f"[NLE] using sole seed dir {subdir.name}", flush=True)
        elif len(hits) > 1:
            sys.exit(f"multiple NLE seed dirs under {base}: "
                     f"{[d.name for d in hits]}. Set nle.seed_subdir=true and "
                     f"nle.init_seed=<N> to pick one.")
    if not (subdir / "model_config.json").exists():
        return None
    model, t_mean, t_std = load_nle(str(subdir), device)
    with open(subdir / "model_config.json") as f:
        n_params = int(json.load(f)["n_params"])
    print(f"[NLE] scope={scope} family={kind} loaded from {subdir}", flush=True)

    tag = f"{scope}_{kind}"
    if subdir.name.startswith("seed_"):
        tag = f"{tag}_{subdir.name}"
    print(f"[NLE] chains tag = {tag}", flush=True)

    return model, n_params, t_mean.astype(np.float32), t_std.astype(np.float32), tag


class LevelMarginalNLE(torch.nn.Module):
    """[cond5] NLE trained on p(t | theta_inferred, nuisance) -> likelihood of the inferred params only ..."""
    def __init__(self, model, levels):
        super().__init__()
        self.model = model
        self.levels = [float(v) for v in levels]

    def log_prob(self, theta, t):
        n, L = theta.shape[0], len(self.levels)
        lv = torch.tensor(self.levels, dtype=theta.dtype, device=theta.device)
        th = torch.cat([theta.repeat(L, 1), lv.repeat_interleave(n)[:, None]], 1)
        lp = self.model.log_prob(th, t.repeat(L, 1)).view(L, n)
        return torch.logsumexp(lp, 0) - math.log(L)


def load_model_legacy(nle_root, device):
    """Old single-file layout: {arm}/nle/best_mdn_nle.pt (GMM only)."""
    from sbi_project_old.sbi.nle_old import MDNLikelihood
    ckpt_path = Path(nle_root) / "best_mdn_nle.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MDNLikelihood(int(ckpt["n_params"]), int(ckpt["n_summaries"]), int(ckpt["n_mix"]),
                          int(ckpt["hidden"]), ckpt.get("param_style", "semelin"),
                          float(ckpt.get("diag_floor", 1e-4))).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    n_params = int(ckpt["n_params"])
    t_mean = t_std = None
    if bool(ckpt.get("t_normalized", False)):
        t_mean = np.load(Path(nle_root) / "t_mean.npy").astype(np.float32)
        t_std = np.load(Path(nle_root) / "t_std.npy").astype(np.float32)
    print(f"[NLE] LEGACY gmm loaded from {ckpt_path} "
          f"(style={ckpt.get('param_style')}, t_normalized={t_mean is not None})", flush=True)
    return model, n_params, t_mean, t_std, "gmm_legacy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--task-num", type=int, default=0)
    ap.add_argument("--task-nb", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batched", action=argparse.BooleanOptionalAction, default=True,
                    help="run all pending targets of this task together (sbi/mcmc_batched.py); "
                         "--no-batched = old one-target-at-a-time run_chain loop")
    ap.add_argument("--batch-targets", type=int, default=320,
                    help="--batched: max targets per batch (memory ~8 MB per target)")
    ap.add_argument("--raw-t", dest="standardize", action="store_false", default=None,
                    help="load the NLE trained on RAW t (from {arm}/nle/raw_t/<model>/).")
    ap.add_argument("--standardize", dest="standardize", action="store_true", default=None,
                    help="load the standardized-t NLE (default).")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    apply_arm_name(cfg)
    nc, mc, dirs = cfg["nle"], cfg["mcmc"], arm_dirs(cfg)
    print(f"[stage3] arm={cfg['arm_name']}  nle={dirs['nle']}", flush=True)
    if args.standardize is not None:
        nc["standardize"] = bool(args.standardize)
    else:
        nc["standardize"] = bool(nc.get("standardize", True))
    device = torch.device(args.device)

    loaded = load_model_new(dirs["nle"], nc, device) or load_model_legacy(dirs["nle"], device)
    if loaded is None:
        sys.exit(f"no trained NLE found: neither {dirs['nle']}/{nc.get('model','gmm')}/"
                 f"model_config.json nor {dirs['nle']}/best_mdn_nle.pt exists. "
                 f"Run stage2_nle.py first (with the matching -o nle.model=...).")
    model, n_params, t_mean, t_std, kind = loaded
    if n_params > int(cfg["n_params"]):
        if n_params != int(cfg["n_params"]) + 1:
            sys.exit(f"[cond5] NLE has {n_params} params, config n_params={cfg['n_params']}: "
                     f"only ONE marginalised nuisance column is supported")
        levels = mc.get("marginalise_levels")
        if not levels:
            sys.exit("[cond5] NLE conditions on an extra param: set mcmc.marginalise_levels "
                     "(e.g. [0.0, 0.5, 1.0] for f_esc)")
        model = LevelMarginalNLE(model, levels).eval()
        print(f"[cond5] NLE has {n_params} params; MCMC over {cfg['n_params']}, extra column "
              f"marginalised over levels {levels}", flush=True)
        n_params = int(cfg["n_params"])
    t_noise = float(nc.get("t_noise_sigma", 0.0))
    _scope = "standard_t" if nc["standardize"] else "raw_t"
    _rest = kind[len(_scope) + 1:] if kind.startswith(_scope + "_") else None
    if _rest is not None:
        _fam, _, _seed = _rest.partition("_")
        _meta = Path(dirs["nle"]) / _scope / _fam / _seed / "run_meta.json" if _seed \
            else Path(dirs["nle"]) / _scope / _fam / "run_meta.json"
        if _meta.exists():
            _trained = float(json.load(open(_meta)).get("t_noise_sigma", 0.0))
            if abs(_trained - t_noise) > 1e-12:
                sys.exit(f"[t-noise] NLE trained with t_noise_sigma={_trained} but config "
                         f"says {t_noise} ({_meta})")
    if t_noise > 0:
        if t_mean is None:
            sys.exit("[t-noise] needs a standardized-t NLE (t_noise_sigma is in standardized units)")
        print(f"[t-noise] adding one N(0, {t_noise}^2) draw to each standardized t_obs "
              f"(seeded by sim, row)", flush=True)

    chains_dir = Path(dirs["root"]) / str(mc.get("chains_dirname", "chains")) / kind
    chains_dir.mkdir(parents=True, exist_ok=True)

    src = load_source(str(dirs["summaries"]), mmap=True)
    theta_np, t_np, sim_ids, noise_ids = src.theta, src.t, src.sim_ids, src.noise_ids

    extra_theta = None
    if n_params > theta_np.shape[1]:
        z = np.load(cfg["data"]["summaries_npz"])
        u, first = np.unique(z["original_sim_ids"], return_index=True)
        extra_theta = (u, z["theta"][first].astype(np.float32))
        del z
        print(f"[cond5] truth: theta cols {theta_np.shape[1]}..{n_params - 1} from summaries_npz",
              flush=True)

    def truth_of(row, sim):
        th = np.asarray(theta_np[row], np.float32)
        if extra_theta is not None:
            u, per_sim = extra_theta
            i = int(np.searchsorted(u, sim)); assert u[i] == sim
            assert np.allclose(per_sim[i, :th.shape[0]], th, atol=1e-6)
            th = per_sim[i]
        return th[:n_params]

    rows = []
    for tg in load_sbc_targets(mc["target_path"]):
        r = np.where(sim_ids == tg["sim"])[0]
        if len(r) and tg["offset"] < len(r):
            rows.append(int(r[tg["offset"]]))
    rows = np.array(rows, np.int64)[args.task_num::args.task_nb]
    print(f"targets this task: {len(rows)} | family={kind} | chains -> {chains_dir}", flush=True)

    def save(base, row, sim, target_t, samples, slogp, accr):
        if samples.shape[0] == 0:
            print(f"empty sim={sim} row={row}", flush=True); return
        samples.tofile(base.with_suffix(".dat"))
        np.save(f"{base}_logp.npy", slogp)
        np.save(f"{base}_truth.npy", truth_of(row, sim))
        np.save(f"{base}_tobs.npy", target_t)
        np.save(f"{base}_meta.npy", np.array([mc["walkers"], mc["steps"], mc["burnin"], n_params,
                                              accr, samples.shape[0], row, sim], np.float32))
        print(f"saved {base.name} {samples.shape} acc={accr:.3f}", flush=True)

    pending = []
    for row in rows:
        row, sim = int(row), int(sim_ids[row])
        nv = np.asarray(noise_ids[row])
        tag = str(int(nv)) if nv.ndim == 0 else "_".join(map(str, nv.astype(int).tolist()))
        arm_tag = cfg["arm_name"].replace("/", "_")
        base = chains_dir / f"SBC_CHAINS_{arm_tag}_sim{sim}_row{row}_noise{tag}"

        if base.with_suffix(".dat").exists() and not args.overwrite:
            print("skip", base.name, flush=True); continue
        target_t = np.asarray(t_np[row], np.float32)
        if t_mean is not None:
            target_t = ((target_t - t_mean) / t_std).astype(np.float32)
        if t_noise > 0:
            eps = np.random.default_rng([int(mc["seed"]), sim, row, 7]).standard_normal(target_t.shape)
            target_t = (target_t + t_noise * eps).astype(np.float32)
        if not args.batched:
            samples, slogp, accr = run_chain(model, target_t, n_params, device, mc, mc["seed"] + sim)
            save(base, row, sim, target_t, samples, slogp, accr)
        else:
            pending.append((base, row, sim, target_t))

    if args.batched and pending:
        from sbi.mcmc_batched import run_chains_batched
        n = max(1, args.batch_targets)
        print(f"[batched] {len(pending)} targets to run, chunks of {n}", flush=True)
        for c in range(0, len(pending), n):
            chunk = pending[c:c + n]
            res = run_chains_batched(model, np.stack([p[3] for p in chunk]), n_params, device, mc,
                                     [mc["seed"] + p[2] for p in chunk])
            for (base, row, sim, target_t), (samples, slogp, accr) in zip(chunk, res):
                save(base, row, sim, target_t, samples, slogp, accr)


if __name__ == "__main__":
    main()
