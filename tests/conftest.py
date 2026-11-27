"""Shared fixtures: src/ on the path, tiny synthetic dataset, a small random NSF."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vmim21cm.synthetic import make_configs, make_data  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """Synthetic data + configs (no runs); -> (dir, {name: config path})."""
    d = str(tmp_path_factory.mktemp("synth"))
    make_data(d)
    return d, make_configs(d)


@pytest.fixture
def small_nsf():
    """Untrained 2-layer NSF on t_dim 3, 4 parameters."""
    import torch
    from vmim21cm.nle import build_nle
    torch.manual_seed(0)
    nc = dict(hidden=8, n_layers=2, num_bins=4, tail_bound=5.0)
    return build_nle(nc, 3, 4).eval(), nc


@pytest.fixture
def mc():
    """Short MCMC settings with the wedge prior on."""
    return dict(walkers=8, steps=12, burnin=4, seed=13, loglik_batch_size=4096,
                use_wedge_prior=True, wedge_tau_idx=1, wedge_mmin_idx=3)


@pytest.fixture
def rng():
    return np.random.default_rng(1)
