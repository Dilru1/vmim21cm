import numpy as np
import torch

from vmim21cm.mcmc import batch_loglik, run_chain, run_chains_batched, sample_prior


def _in_wedge(x, ti=1, mi=3):
    return (x[:, mi] <= 1.98 - 1.88 * x[:, ti]) & (x[:, mi] >= 0.90 - 1.88 * x[:, ti])


def test_prior_samples_respect_box_and_wedge(rng):
    x = sample_prior(rng, 5000, 4, True, 1, 3)
    assert x.shape == (5000, 4) and np.all((x >= 0) & (x <= 1)) and np.all(_in_wedge(x))
    y = sample_prior(rng, 100, 2, True, 1, 3)                    # wedge ignored for 2 params
    assert y.shape == (100, 2)


def test_loglik_is_minus_inf_outside_prior(small_nsf):
    model, _ = small_nsf
    theta = np.array([[0.5, 0.3, 0.5, 0.8],      # inside
                      [0.5, 0.3, 0.5, 0.1],      # below the wedge
                      [1.2, 0.3, 0.5, 0.8]],     # outside the box
                     np.float32)
    lp = batch_loglik(model, theta, np.zeros(3, np.float32), torch.device("cpu"), 2, True, 1, 3)
    assert np.isfinite(lp[0]) and np.isneginf(lp[1]) and np.isneginf(lp[2])


def test_chain_outputs(small_nsf, mc):
    model, _ = small_nsf
    s, lp, acc = run_chain(model, np.zeros(3, np.float32), 4, torch.device("cpu"), mc, seed=13)
    assert s.shape == ((mc["steps"] - mc["burnin"]) * mc["walkers"], 4) and lp.shape == (len(s),)
    assert np.all(np.isfinite(lp)) and 0.0 <= acc <= 1.0 and np.all(_in_wedge(s))


def test_batched_equals_single_chains(small_nsf, mc):
    model, _ = small_nsf
    targets = np.random.default_rng(3).standard_normal((3, 3)).astype(np.float32)
    seeds = [13 + 5, 13 + 9, 13 + 40]
    batched = run_chains_batched(model, targets, 4, torch.device("cpu"), mc, seeds, log_every=0)
    for (bs, blp, bacc), t, sd in zip(batched, targets, seeds):
        s, lp, acc = run_chain(model, t, 4, torch.device("cpu"), mc, sd)
        assert s.shape == bs.shape and abs(acc - bacc) < 1e-12
        assert np.allclose(s, bs, atol=1e-5) and np.allclose(lp, blp, atol=1e-4)
