import numpy as np
import pytest
import torch

from vmim21cm.cnn import CubeCompressor
from vmim21cm.compressors import CNNCompressor, GMMHead, MLPCompressor


def test_mlp_shapes_and_direct():
    m = MLPCompressor(30, 8, 4, [32, 16])
    t, aux = m(torch.randn(5, 30))
    assert t.shape == (5, 8) and aux.shape == (5, 4)
    d = MLPCompressor(30, 4, 4, [16], direct=True)
    t, aux = d(torch.randn(5, 30))
    assert t is aux
    with pytest.raises(ValueError):
        MLPCompressor(30, 8, 4, [16], direct=True)


def test_cnn_shapes():
    m = CNNCompressor(3, 8, 4).eval()
    t, aux = m(torch.randn(2, 3, 16, 16, 16))
    assert t.shape == (2, 8) and aux.shape == (2, 4)
    with pytest.raises(ValueError):
        CNNCompressor(3, 8, 4, direct=True)


def test_gmm_head_density():
    torch.manual_seed(0)
    h = GMMHead(t_dim=1, n_params=1, n_mix=2, hidden=8, sigma_floor=1e-2)
    grid = torch.linspace(-6, 7, 4001)[:, None]
    lp = h.log_prob(torch.zeros(len(grid), 1), grid)
    integral = torch.trapezoid(lp.exp(), grid[:, 0])
    assert abs(float(integral) - 1.0) < 1e-3                    # normalised density
    assert torch.all(h.conditional_sigma(torch.zeros(3, 1)) > 0)


def test_gmm_head_sigma_floor():
    h = GMMHead(2, 3, 2, 8, sigma_floor=0.05)
    with torch.no_grad():
        for p in h.raw_tril.parameters():
            p.fill_(-50.0)                                      # softplus -> 0
    L = h._params(torch.zeros(4, 2))[2]
    assert torch.allclose(torch.diagonal(L, dim1=-2, dim2=-1), torch.full((4, 2, 3), 0.05))


@pytest.mark.parametrize("direct", [False, True])
def test_cube_compressor_matches_training_path(tmp_path, rng, direct):
    torch.manual_seed(0)
    model = CNNCompressor(2, 4, 4, direct=direct).eval()
    torch.save(model.state_dict(), tmp_path / "compressor.pt")
    means, stds = np.zeros(2, np.float32), np.array([1.5, 0.7], np.float32)
    np.savez(tmp_path / "cube_stats.npz", means=means, stds=stds, noise_scale=1.0)

    cc = CubeCompressor.from_run(tmp_path)
    assert cc.t_dim == 4 and cc.model.direct == direct
    cubes = rng.standard_normal((3, 2, 16, 16, 16)).astype(np.float32) + 2.0
    x = np.stack([np.stack([(c - c.mean() - means[k]) / (stds[k] + 1e-8) for k, c in enumerate(cube)])
                  for cube in cubes]).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))[0].numpy()
    assert np.allclose(cc.compress(cubes, batch_size=2), ref, atol=1e-5)
    assert np.allclose(cc(cubes[0]), ref[0], atol=1e-5)


def test_cube_compressor_needs_stats(tmp_path):
    torch.save(CNNCompressor(1, 4, 4).state_dict(), tmp_path / "compressor.pt")
    with pytest.raises(FileNotFoundError):
        CubeCompressor.from_run(tmp_path)
