import numpy as np
import torch

from vmim21cm.nle import load_nle, save_nle


def test_log_prob_shape_and_finite(small_nsf):
    model, _ = small_nsf
    lp = model.log_prob(torch.rand(10, 4), torch.randn(10, 3))
    assert lp.shape == (10,) and torch.all(torch.isfinite(lp))


def test_flow_is_normalised_in_1d():
    from vmim21cm.nle import build_nle
    torch.manual_seed(0)
    m = build_nle(dict(hidden=8, n_layers=2, num_bins=4, tail_bound=5.0), 1, 2).eval()
    grid = torch.linspace(-12, 12, 6001)[:, None]
    with torch.no_grad():
        lp = m.log_prob(torch.full((len(grid), 2), 0.3), grid)
    assert abs(float(torch.trapezoid(lp.exp(), grid[:, 0])) - 1.0) < 1e-3


def test_save_load_roundtrip(tmp_path, small_nsf):
    model, nc = small_nsf
    save_nle(model, str(tmp_path), nc, 3, 4, np.ones(3, np.float32), 2 * np.ones(3, np.float32))
    m2, P, mu, sd = load_nle(str(tmp_path))
    th, t = torch.rand(6, 4), torch.randn(6, 3)
    with torch.no_grad():
        assert torch.equal(model.log_prob(th, t), m2.log_prob(th, t))
    assert P == 4 and np.all(mu == 1) and np.all(sd == 2)
