import numpy as np

from vmim21cm.data import load_sbc_targets, load_summaries, split_by_sim


def test_split_by_sim_is_disjoint_and_complete(rng):
    sims = np.repeat(np.arange(30), 5)
    theta, t = rng.random((150, 4)), rng.random((150, 6))
    noise = np.tile(np.arange(5), 30)
    out = split_by_sim(theta, t, sims, noise, 0.2, seed=42)
    sim_tr, sim_va, train, val = out[2], out[6], out[8], out[9]
    assert set(train).isdisjoint(val) and len(train) + len(val) == 30 and len(val) == 6
    assert set(sim_tr) == set(train) and set(sim_va) == set(val)
    assert len(sim_tr) + len(sim_va) == 150


def test_split_rows_stay_aligned(rng):
    sims = np.repeat(np.arange(10), 3)
    theta = sims[:, None].astype(float) * np.ones((1, 2))
    t = sims[:, None].astype(float) * 10
    th_tr, t_tr, s_tr = split_by_sim(theta, t, sims, np.zeros(30), 0.3, seed=0)[:3]
    assert np.all(th_tr[:, 0] == s_tr) and np.all(t_tr[:, 0] == 10 * s_tr)


def test_split_is_reproducible():
    sims = np.repeat(np.arange(20), 2)
    c = split_by_sim(np.arange(40.0)[:, None], np.arange(40.0)[:, None], sims, np.zeros(40), 0.1, seed=7)
    d = split_by_sim(np.arange(40.0)[:, None], np.arange(40.0)[:, None], sims, np.zeros(40), 0.1, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(c, d))


def test_load_summaries_npz_and_dir(tmp_path, rng):
    theta, t, sims = rng.random((6, 4)), rng.random((6, 3)), np.repeat([0, 1], 3)
    np.savez(tmp_path / "s.npz", theta=theta, t=t, original_sim_ids=sims)
    th, tt, s, nz = load_summaries(tmp_path / "s.npz")
    assert th.dtype == np.float32 and tt.shape == (6, 3) and np.all(nz == 0)
    d = tmp_path / "summaries"
    d.mkdir()
    for k, v in dict(theta=theta, t=t, original_sim_ids=sims, noise_ids=np.arange(6)).items():
        np.save(d / f"{k}.npy", v)
    th2, tt2, s2, nz2 = load_summaries(d)
    assert np.allclose(th2, th) and np.array_equal(nz2, np.arange(6))


def test_load_sbc_targets(tmp_path):
    arr = np.array([{"sim": 3, "offset": 1}, {"sim": np.int64(5), "offset": 0}], dtype=object)
    np.save(tmp_path / "t.npy", arr, allow_pickle=True)
    assert load_sbc_targets(tmp_path / "t.npy") == [{"sim": 3, "offset": 1}, {"sim": 5, "offset": 0}]
