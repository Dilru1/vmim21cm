"""Tiny synthetic dataset and configs (all three pipelines) for tests and a first local run."""
import os

import numpy as np
import yaml


def make_data(d, n_sims=48, n_noise=6, box=16):
    """Write summaries.npz, two-channel signal/noise cubes, params, sim ids and SBC targets to d."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    theta = rng.random((n_sims, 4)).astype(np.float32)
    t = np.repeat(np.concatenate([theta, theta ** 2], 1), n_noise, 0)
    t = (t + 0.05 * rng.standard_normal(t.shape)).astype(np.float32)
    np.savez(f"{d}/summaries.npz", theta=np.repeat(theta, n_noise, 0), t=t,
             original_sim_ids=np.repeat(np.arange(n_sims), n_noise))
    phys = np.stack([10 ** (2 * theta[:, 0] - 1) * 0.00192, theta[:, 1], theta[:, 2], 8 + 1.6 * theta[:, 3]], 1)
    np.save(f"{d}/params.npy", phys.astype(np.float32))
    np.save(f"{d}/sim_ids.npy", np.arange(n_sims))
    for c in range(2):
        g = rng.standard_normal((n_sims, box, box, box))
        sig = g * (1 + theta[:, c, None, None, None]) + theta[:, 2 + c, None, None, None] * g ** 2
        sig.astype(np.float64).tofile(f"{d}/sig{c}.dat")
        rng.standard_normal((20, box, box, box)).astype(np.float64).tofile(f"{d}/noise{c}.dat")
    targets = np.array([{"sim": int(s), "offset": 0} for s in range(0, n_sims, 3)], dtype=object)
    np.save(f"{d}/targets.npy", targets, allow_pickle=True)


def make_configs(d, n_sims=48, n_noise=6, box=16):
    """Write baseline/patha/pathb YAMLs and eval.yaml for the synthetic data; -> {name: path}."""
    nle = dict(split_seed=42, val_frac=0.2, nsplit=2, batch_size=32, lr_phase1=1e-3, lr_phase2=1e-4,
               num_workers=0, hidden=16, n_layers=2, num_bins=4, tail_bound=5.0)
    mcmc = dict(target_path=f"{d}/targets.npy", walkers=16, steps=60, burnin=20, seed=13,
                loglik_batch_size=1024, use_wedge_prior=False, wedge_tau_idx=1, wedge_mmin_idx=3)
    common = dict(out_root=f"{d}/out", n_params=4, nle=nle, mcmc=mcmc)
    train = dict(lr=1e-3, epochs=5, plateau_patience=5, num_workers=0, split_seed=43, val_frac=0.2,
                 probe_every=5)
    base = dict(common, name="baseline", seeds={"nle": [42]}, data={"summaries_npz": f"{d}/summaries.npz"},
                compressor={"type": "none"})
    patha = dict(common, name="patha", seeds={"compressor": [43], "nle": [42]},
                 data={"summaries_npz": f"{d}/summaries.npz"},
                 compressor=dict(train, type="mlp", objective="vmim", t_dim=4, dense_layers=[32, 32],
                                 dropout=0.0, n_mix_head=2, hidden_head=16, sigma_floor=1e-2,
                                 head_lr_mult=5.0, batch_size=8, save_batch_size=64))
    pathb = dict(common, name="pathb", seeds={"compressor": [43], "nle": [42]},
                 data=dict(box=box, n_signal=n_sims, n_noise=n_noise,
                           signal_paths=[f"{d}/sig0.dat", f"{d}/sig1.dat"],
                           noise_paths=[f"{d}/noise0.dat", f"{d}/noise1.dat"], params_path=f"{d}/params.npy",
                           sim_ids_path=f"{d}/sim_ids.npy",
                           params=[{"transform": "log10", "scale": 0.00192, "log_lo": -1.0, "log_hi": 1.0},
                                   {"transform": "none"}, {"transform": "none"},
                                   {"transform": "linear", "lo": 8.0, "hi": 9.6}]),
                 compressor=dict(train, type="cnn", objective="mse", lam_mse=1.0, t_dim=4, noise_scale=1.0,
                                 batch_size=8, save_batch_size=32))
    paths = {}
    for c in (base, patha, pathb):
        paths[c["name"]] = f"{d}/{c['name']}.yaml"
        with open(paths[c["name"]], "w") as f:
            yaml.safe_dump(c, f)
    ev = dict(out=f"{d}/out/eval", thin=1, dlogp=10.0, burnin_frac=0.0, nbins=4, walker_cut=5.0,
              params=[0, 1, 2, 3], param_names=["p0", "p1", "p2", "p3"],
              groups=[{"label": k, "config": v} for k, v in paths.items()])
    with open(f"{d}/eval.yaml", "w") as f:
        yaml.safe_dump(ev, f)
    return paths


def add_args(ap):
    pass


def run(args):
    d = os.path.abspath(args.config)
    make_data(d)
    p = make_configs(d)
    print(f"synthetic data + configs in {d}\n")
    for name in ("baseline", "patha", "pathb"):
        print(f"  vmim21cm run {p[name]} --device cpu")
    print(f"  vmim21cm stage4 {d}/eval.yaml")
