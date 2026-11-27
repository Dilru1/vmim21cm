import numpy as np

from vmim21cm.evaluate import chain_sim, eval_run, half_width, read_chain, walker_mask

EV = dict(walker_cut=5.0, burnin_frac=0.0, thin=1, dlogp=1e9, nbins=4)


def _write_chain(d, sim, samples, truth, logp=None, walkers=None, steps=None):
    base = d / f"chain_sim{sim}_row{sim * 10}"
    samples.astype(np.float32).tofile(f"{base}.dat")
    np.save(f"{base}_truth.npy", truth.astype(np.float32))
    if logp is not None:
        np.save(f"{base}_logp.npy", logp.astype(np.float32))
        np.save(f"{base}_meta.npy", np.array([walkers, steps, 0, samples.shape[1], 0.3, len(samples), 0, sim],
                                             np.float32))
    return f"{base}.dat"


def test_chain_sim_and_half_width():
    assert chain_sim("/a/chain_sim123_row9.dat") == 123 and chain_sim("x.dat") is None
    x = np.random.default_rng(0).standard_normal(200000)
    assert abs(half_width(x) - 1.0) < 0.01


def test_walker_cut_drops_stuck_walker(tmp_path):
    W, S = 10, 30
    logp = np.zeros((S, W))
    logp[:, 3] = -100.0                                          # one stuck walker
    f = _write_chain(tmp_path, 1, np.zeros((S * W, 2)), np.zeros(2), logp.reshape(-1), W, S)
    m = walker_mask(f[:-4], S * W, 5.0)
    assert m.sum() == S * (W - 1) and not m.reshape(S, W)[:, 3].any()


def test_read_chain_rejects_bad_files(tmp_path):
    f = _write_chain(tmp_path, 2, np.zeros((10, 2)), np.zeros(2))          # < 20 samples
    assert read_chain(f, 2, EV) is None
    f = _write_chain(tmp_path, 3, np.zeros((50, 2)), np.zeros(2))
    assert read_chain(f, 3, EV) is None                                   # size not a multiple


def test_calibrated_posteriors_give_ratio_near_one(tmp_path):
    rng = np.random.default_rng(0)
    sims = range(400)
    for s in sims:                                   # truth drawn from the posterior itself
        post = rng.normal(0.5, 0.1, size=(200, 2))
        _write_chain(tmp_path, s, post, rng.normal(0.5, 0.1, size=2))
    r = eval_run(str(tmp_path), 2, [0, 1], EV, set(sims))
    assert r["used"] == 400
    assert np.allclose(np.mean(r["rank_hist"], axis=1), 1.0)
    assert all(c < 3.0 for c in r["calib_ratio"])               # chi^2 / dof-like, ~1 when calibrated
    assert abs(r["sigma"][0] - 0.1) < 0.01 and abs(r["gv"] - 0.01) < 0.002


def test_overconfident_posteriors_are_flagged(tmp_path):
    rng = np.random.default_rng(1)
    for s in range(400):                             # posteriors 4x too narrow
        post = rng.normal(0.5, 0.025, size=(200, 1))
        _write_chain(tmp_path, s, post, rng.normal(0.5, 0.1, size=1))
    r = eval_run(str(tmp_path), 1, [0], EV, set(range(400)))
    assert r["calib_ratio"][0] > 10
