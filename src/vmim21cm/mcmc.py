"""Stage-3 stretch-move ensemble sampler, uniform [0,1]^P prior and optional (tau, Mmin) wedge."""

import math
import numpy as np
import torch


@torch.no_grad()
def batch_loglik(model, theta, target_t, device, bs, use_prior, ti, mi):
    """log q(t_obs | theta) for every row of theta, -inf outside the prior."""
    theta = np.asarray(theta, np.float32); target_t = np.asarray(target_t, np.float32)
    out = np.empty(theta.shape[0], np.float32)
    for s in range(0, theta.shape[0], bs):
        e = min(s + bs, theta.shape[0])
        tt = torch.from_numpy(np.repeat(target_t[None, :], e - s, axis=0)).to(device)
        out[s:e] = model.log_prob(torch.from_numpy(theta[s:e]).to(device), tt).cpu().numpy()
    mask = np.any(theta < 0.0, axis=-1) | np.any(theta > 1.0, axis=-1)
    if use_prior and theta.shape[1] > max(ti, mi):
        mask |= (theta[:, mi] > (1.98 - 1.88 * theta[:, ti])) | (theta[:, mi] < (0.90 - 1.88 * theta[:, ti]))
    return np.where(mask, -np.inf, out)


def sample_prior(rng, n, n_params, use_prior, ti, mi):
    """n prior draws (uniform, wedge 0.90 - 1.88 tau <= Mmin <= 1.98 - 1.88 tau if use_prior)."""
    got = []
    while sum(len(x) for x in got) < n:
        x = rng.random((max(1000, 5 * n), n_params), dtype=np.float32)
        ok = np.all((x >= 0.0) & (x <= 1.0), axis=1)
        if use_prior and n_params > max(ti, mi):
            ok &= x[:, mi] <= (1.98 - 1.88 * x[:, ti])
            ok &= x[:, mi] >= (0.90 - 1.88 * x[:, ti])
        got.append(x[ok])
    return np.concatenate(got, 0)[:n]


def run_chain(model, target_t, n_params, device, mc, seed):
    """One target -> (post-burn-in samples [step*walkers + walker], log p, acceptance)."""
    rng, pool = np.random.default_rng(seed), mc["walkers"] // 2
    steps, walkers, burnin = mc["steps"], mc["walkers"], mc["burnin"]
    up, ti, mi, bs = mc["use_wedge_prior"], mc["wedge_tau_idx"], mc["wedge_mmin_idx"], mc["loglik_batch_size"]
    chain = np.zeros((steps, walkers, n_params), np.float32)
    logp = np.zeros((steps, walkers), np.float32)
    chain[0] = sample_prior(rng, walkers, n_params, up, ti, mi)
    logp[0] = batch_loglik(model, chain[0], target_t, device, bs, up, ti, mi)
    sa = math.sqrt(2.0); acc = prop = 0
    for step in range(steps - 1):
        if step % 50 == 0:
            fin = logp[step][np.isfinite(logp[step])]
            print(f"  step {step}/{steps-1} best {np.max(fin) if len(fin) else -np.inf:.3f}", flush=True)
        npos, nlogp = chain[step].copy(), logp[step].copy()
        for sub in range(2):
            lo, hi = pool * sub, pool * (sub + 1)
            partner = rng.integers(pool, size=pool) + pool * ((sub + 1) % 2)
            z = (rng.random(pool) * (sa - 1.0 / sa) + 1.0 / sa) ** 2
            proposal = chain[step, partner] + z[:, None] * (chain[step, lo:hi] - chain[step, partner])
            pl = batch_loglik(model, proposal, target_t, device, bs, up, ti, mi)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lar = np.nan_to_num((n_params - 1) * np.log(z) + pl - logp[step, lo:hi],
                                    nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            a = np.log(rng.random(pool)) < lar
            acc += int(a.sum()); prop += pool
            npos[lo:hi] = np.where(a[:, None], proposal, chain[step, lo:hi])
            nlogp[lo:hi] = np.where(a, pl, logp[step, lo:hi])
        chain[step + 1], logp[step + 1] = npos, nlogp
    pc, pl = chain[burnin:].reshape(-1, n_params), logp[burnin:].reshape(-1)
    keep = np.isfinite(pl)
    return pc[keep].astype(np.float32), pl[keep].astype(np.float32), float(acc / max(prop, 1))


@torch.no_grad()
def _loglik_multi(model, theta, tt, device, bs, use_prior, ti, mi):
    """batch_loglik with a separate observed t for every row of theta."""
    out = np.empty(theta.shape[0], np.float32)
    for s in range(0, theta.shape[0], bs):
        e = min(s + bs, theta.shape[0])
        out[s:e] = model.log_prob(torch.from_numpy(theta[s:e]).to(device),
                                  torch.from_numpy(tt[s:e]).to(device)).cpu().numpy()
    mask = np.any(theta < 0.0, axis=-1) | np.any(theta > 1.0, axis=-1)
    if use_prior and theta.shape[1] > max(ti, mi):
        mask |= (theta[:, mi] > (1.98 - 1.88 * theta[:, ti])) | (theta[:, mi] < (0.90 - 1.88 * theta[:, ti]))
    return np.where(mask, -np.inf, out)


def run_chains_batched(model, targets_t, n_params, device, mc, seeds, log_every=100):
    """run_chain for S targets in one likelihood call per half-move; same chains, ~10x cheaper."""
    S = len(seeds)
    steps, walkers, burnin = mc["steps"], mc["walkers"], mc["burnin"]
    pool = walkers // 2
    up, ti, mi = mc["use_wedge_prior"], mc["wedge_tau_idx"], mc["wedge_mmin_idx"]
    bs = int(max(mc["loglik_batch_size"], S * pool))
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    T = np.asarray(targets_t, np.float32)
    tt_all, tt_half = np.repeat(T, walkers, axis=0), np.repeat(T, pool, axis=0)

    chain = np.zeros((steps, S, walkers, n_params), np.float32)
    logp = np.zeros((steps, S, walkers), np.float32)
    chain[0] = np.stack([sample_prior(r, walkers, n_params, up, ti, mi) for r in rngs])
    logp[0] = _loglik_multi(model, chain[0].reshape(-1, n_params), tt_all, device, bs,
                            up, ti, mi).reshape(S, walkers)
    sa = math.sqrt(2.0)
    acc, prop = np.zeros(S, np.int64), 0
    ar = np.arange(S)[:, None]
    for step in range(steps - 1):
        if log_every and step % log_every == 0:
            print(f"  [batched S={S}] step {step}/{steps-1}", flush=True)
        npos, nlogp = chain[step].copy(), logp[step].copy()
        for sub in range(2):
            lo, hi = pool * sub, pool * (sub + 1)
            off = pool * ((sub + 1) % 2)
            partner = np.empty((S, pool), np.int64)
            z = np.empty((S, pool), np.float64)
            for i, r in enumerate(rngs):
                partner[i] = r.integers(pool, size=pool) + off
                z[i] = (r.random(pool) * (sa - 1.0 / sa) + 1.0 / sa) ** 2
            cur, prt = chain[step][:, lo:hi], chain[step][ar, partner]
            proposal = (prt + z[..., None] * (cur - prt)).astype(np.float32)
            pl = _loglik_multi(model, proposal.reshape(-1, n_params), tt_half, device, bs,
                               up, ti, mi).reshape(S, pool)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lar = np.nan_to_num((n_params - 1) * np.log(z) + pl - logp[step][:, lo:hi],
                                    nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            a = np.stack([np.log(r.random(pool)) for r in rngs]) < lar
            acc += a.sum(1); prop += pool
            npos[:, lo:hi] = np.where(a[..., None], proposal, cur)
            nlogp[:, lo:hi] = np.where(a, pl, logp[step][:, lo:hi])
        chain[step + 1], logp[step + 1] = npos, nlogp
    out = []
    for i in range(S):
        pc, pli = chain[burnin:, i].reshape(-1, n_params), logp[burnin:, i].reshape(-1)
        keep = np.isfinite(pli)
        out.append((pc[keep].astype(np.float32), pli[keep].astype(np.float32),
                    float(acc[i] / max(prop, 1))))
    return out
