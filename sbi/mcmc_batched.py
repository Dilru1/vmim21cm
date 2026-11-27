"""Batched SBC MCMC: the stretch-move sampler of sbi/mcmc.py run for MANY targets at once."""
import math
import numpy as np
import torch
from sbi.mcmc import sample_prior


@torch.no_grad()
def _loglik_multi(model, theta, tt, device, bs, use_prior, ti, mi):
    out = np.empty(theta.shape[0], np.float32)
    for s in range(0, theta.shape[0], bs):
        e = min(s + bs, theta.shape[0])
        out[s:e] = model.log_prob(torch.from_numpy(theta[s:e]).to(device),
                                  torch.from_numpy(tt[s:e]).to(device)).cpu().numpy()
    mask = np.any(theta < 0.0, axis=-1) | np.any(theta > 1.0, axis=-1)
    if use_prior and theta.shape[1] > max(ti, mi):
        mask |= (theta[:, mi] > (1.98 - 1.88 * theta[:, ti])) | (theta[:, mi] < (0.90 - 1.88 * theta[:, ti]))
    return np.where(mask, -np.inf, out)


def run_chains_batched(model, targets_t, n_params, device, mc, seeds, bs=None, log_every=100):
    """targets_t (S, k) float32, seeds length S (= mc['seed'] + sim, as in stage 3)."""
    import time
    S = len(seeds)
    steps, walkers, burnin = mc["steps"], mc["walkers"], mc["burnin"]
    pool = walkers // 2
    up, ti, mi = mc["use_original_prior"], mc["wedge_tau_idx"], mc["wedge_mmin_idx"]
    bs = int(bs or max(mc["loglik_batch_size"], S * pool))
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    T = np.asarray(targets_t, np.float32)
    tt_all = np.repeat(T, walkers, axis=0)
    tt_half = np.repeat(T, pool, axis=0)

    chain = np.zeros((steps, S, walkers, n_params), np.float32)
    logp = np.zeros((steps, S, walkers), np.float32)
    chain[0] = np.stack([sample_prior(r, walkers, n_params, up, ti, mi) for r in rngs])
    logp[0] = _loglik_multi(model, chain[0].reshape(-1, n_params), tt_all, device, bs, up, ti, mi
                            ).reshape(S, walkers)
    sa = math.sqrt(2.0)
    acc = np.zeros(S, np.int64); prop = 0
    ar = np.arange(S)[:, None]
    t0 = time.time()
    for step in range(steps - 1):
        if log_every and step % log_every == 0:
            el = time.time() - t0
            eta = el / step * (steps - 1 - step) if step else float("nan")
            print(f"  [batched S={S}] step {step}/{steps-1}  {el:.0f}s elapsed  eta {eta:.0f}s", flush=True)
        npos, nlogp = chain[step].copy(), logp[step].copy()
        for sub in range(2):
            lo, hi = pool * sub, pool * (sub + 1)
            off = pool * ((sub + 1) % 2)
            partner = np.empty((S, pool), np.int64); z = np.empty((S, pool), np.float64)
            for i, r in enumerate(rngs):
                partner[i] = r.integers(pool, size=pool) + off
                z[i] = (r.random(pool) * (sa - 1.0 / sa) + 1.0 / sa) ** 2
            cur = chain[step][:, lo:hi]
            prt = chain[step][ar, partner]
            proposal = (prt + z[..., None] * (cur - prt)).astype(np.float32)
            pl = _loglik_multi(model, proposal.reshape(-1, n_params), tt_half, device, bs, up, ti, mi
                               ).reshape(S, pool)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lar = np.nan_to_num((n_params - 1) * np.log(z) + pl - logp[step][:, lo:hi],
                                    nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            u = np.stack([np.log(r.random(pool)) for r in rngs])
            a = u < lar
            acc += a.sum(1); prop += pool
            npos[:, lo:hi] = np.where(a[..., None], proposal, cur)
            nlogp[:, lo:hi] = np.where(a, pl, logp[step][:, lo:hi])
        chain[step + 1], logp[step + 1] = npos, nlogp
    out = []
    for i in range(S):
        pc = chain[burnin:, i].reshape(-1, n_params); pli = logp[burnin:, i].reshape(-1)
        keep = np.isfinite(pli)
        if mc["filter_dlogp"] is not None and np.any(keep):
            keep &= pli > np.max(pli[keep]) - mc["filter_dlogp"]
        out.append((pc[keep].astype(np.float32), pli[keep].astype(np.float32),
                    float(acc[i] / max(prop, 1))))
    return out
