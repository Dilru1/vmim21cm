"""Stage 3: MCMC posterior for every SBC target with the stage-2 likelihood (CPU)."""
import os

import numpy as np
import torch

from ..config import load_config, run_dir, summaries_path
from ..data import load_sbc_targets, load_summaries
from ..mcmc import run_chain, run_chains_batched
from ..nle import load_nle


def add_args(ap):
    ap.add_argument("--cseed", type=int, default=0)
    ap.add_argument("--nseed", type=int, required=True)
    ap.add_argument("--task", type=int, default=0, help="this job's share of the targets")
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--no-batched", action="store_true", help="one chain at a time (run_chain)")
    ap.add_argument("--batch-targets", type=int, default=320, help="targets per batched chunk (~8 MB each)")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: SLURM_CPUS_PER_TASK, else 4)")


def run(args):
    torch.set_num_threads(args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = load_config(args.config, args.set)
    mc = cfg["mcmc"]
    rd = run_dir(cfg, args.cseed)
    out = rd / f"chains_n{args.nseed}"
    out.mkdir(parents=True, exist_ok=True)
    model, P, t_mean, t_std = load_nle(str(rd / f"nle_n{args.nseed}"))

    # target k = row `offset` of simulation `sim` in the summary table
    theta, t, sim_ids, _ = load_summaries(summaries_path(cfg, args.cseed))
    rows = []
    for tg in load_sbc_targets(mc["target_path"]):
        r = np.where(sim_ids == tg["sim"])[0]
        if len(r) and tg["offset"] < len(r):
            rows.append(int(r[tg["offset"]]))
    rows = np.array(rows, np.int64)[args.task::args.ntasks]
    print(f"stage3 {cfg['name']} c{args.cseed} n{args.nseed}: {len(rows)} targets -> {out}", flush=True)

    def save(row, sim, t_obs, samples, logp, acc):
        # meta = [walkers, steps, burnin, n_params, acceptance, n_samples, row, sim]
        base = out / f"chain_sim{sim}_row{row}"
        if samples.shape[0] == 0:
            print(f"empty sim={sim} row={row}", flush=True)
            return
        np.save(f"{base}_logp.npy", logp)
        np.save(f"{base}_truth.npy", np.asarray(theta[row], np.float32))
        np.save(f"{base}_tobs.npy", t_obs)
        np.save(f"{base}_meta.npy", np.array([mc["walkers"], mc["steps"], mc["burnin"], P, acc,
                                              samples.shape[0], row, sim], np.float32))
        samples.tofile(base.with_suffix(".dat"))      # last: a .dat means the chain is complete
        print(f"saved {base.name} {samples.shape} acc={acc:.3f}", flush=True)

    todo = []
    for row in rows:
        row, sim = int(row), int(sim_ids[row])
        if (out / f"chain_sim{sim}_row{row}.dat").exists():
            print(f"skip chain_sim{sim}_row{row}", flush=True)
            continue
        t_obs = ((np.asarray(t[row], np.float32) - t_mean) / t_std).astype(np.float32)
        todo.append((row, sim, t_obs))
    dev = torch.device("cpu")

    if args.no_batched:
        for row, sim, t_obs in todo:
            save(row, sim, t_obs, *run_chain(model, t_obs, P, dev, mc, mc["seed"] + sim))
        return
    for c in range(0, len(todo), args.batch_targets):
        chunk = todo[c:c + args.batch_targets]
        print(f"batched chunk {c // args.batch_targets + 1}: {len(chunk)} targets", flush=True)
        res = run_chains_batched(model, np.stack([x[2] for x in chunk]), P, dev, mc,
                                 [mc["seed"] + x[1] for x in chunk])
        for (row, sim, t_obs), r in zip(chunk, res):
            save(row, sim, t_obs, *r)
