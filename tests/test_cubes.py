import numpy as np
import pytest
import torch

from vmim21cm.config import load_config
from vmim21cm.cubes import (CubeDataset, ExportTable, _lattice_flip, augment, channel_stats,
                            normalise_params, remove_cube_mean)


def test_normalise_params_transforms():
    p = np.array([[0.00192, 738.9, 0.3, 8.0], [0.0192, 10504.7, 0.7, 9.6]], np.float64)
    spec = [{"transform": "log10", "scale": 0.00192, "log_lo": -1.0, "log_hi": 1.0},
            {"transform": "log10", "lo": 738.9, "hi": 10504.7},
            {"transform": "none"},
            {"transform": "linear", "lo": 8.0, "hi": 9.6}]
    out = normalise_params(p, spec)
    assert np.allclose(out, [[0.5, 0.0, 0.3, 0.0], [1.0, 1.0, 0.7, 1.0]])
    assert np.array_equal(p[:, 2], [0.3, 0.7])                  # input untouched
    with pytest.raises(ValueError):
        normalise_params(p, [{"transform": "sqrt"}])


def test_remove_cube_mean(rng):
    x = rng.standard_normal((3, 2, 8, 8, 8)) + 5.0
    assert np.allclose(remove_cube_mean(x).mean(axis=(-3, -2, -1)), 0.0, atol=1e-12)


def test_lattice_flip_is_an_involution_fixing_the_origin():
    x = torch.randn(2, 6, 6, 6)
    for ax in (1, 2, 3):
        y = _lattice_flip(x, ax)
        assert torch.equal(_lattice_flip(y, ax), x)
        assert torch.equal(y.select(ax, 0), x.select(ax, 0))
        assert torch.equal(y.select(ax, 1), x.select(ax, 5))


def test_augment_permutes_voxels():
    np.random.seed(0)
    x = torch.randn(2, 8, 8, 8)
    for _ in range(5):
        y = augment(x, 8)
        assert y.shape == x.shape
        for c in range(2):
            assert torch.equal(torch.sort(y[c].flatten())[0], torch.sort(x[c].flatten())[0])


def test_dataset_and_export_table(synth):
    d, paths = synth
    cfg = load_config(paths["pathb"])
    dd = cfg["data"]
    means, stds = channel_stats(dd, np.load(dd["sim_ids_path"]).astype(int), 1.0)
    ds = CubeDataset(dd, 4, means, stds, 1.0, do_augment=False)
    x, theta, sid = ds[3]
    assert x.shape == (2, 16, 16, 16) and theta.shape == (4,) and sid == 3
    assert torch.all((theta >= 0) & (theta <= 1))
    assert torch.allclose(x.mean(dim=(1, 2, 3)), torch.zeros(2), atol=1e-5)

    full = ExportTable(ds, n_noise=6)
    assert len(full) == 48 * 6
    a = full[7]
    b = full[7]
    assert torch.equal(a[0], b[0])                                # deterministic noise draws
    part = ExportTable(ds, n_noise=6, start=3, end=6)
    # row (sim 1, rep 4) of the full table = row (sim 1, rep 4 - 3) of the shard
    assert torch.equal(full[1 * 6 + 4][0], part[1 * 3 + 1][0])
    assert torch.equal(full[1 * 6 + 4][3], part[1 * 3 + 1][3])
