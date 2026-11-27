import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def worker_init_fn(worker_id):
    process_seed = torch.initial_seed() % (2**32)
    np.random.seed(process_seed + worker_id)


_MR_MODES = ("none", "cube", "slice")


def _check_mr(mode):
    mode = "none" if mode in (None, False) else str(mode).lower()
    if mode not in _MR_MODES:
        raise ValueError(f"mean_remove must be one of {_MR_MODES}, got {mode!r}")
    return mode


def remove_mean(x, mode):
    """x: single cube (D,H,W) or batch (N,D,H,W)."""
    mode = _check_mr(mode)
    if mode == "none":
        return x
    off = 1 if x.ndim == 4 else 0
    if mode == "cube":
        axes = tuple(range(off, off + 3))
    else:
        axes = (off + 1, off + 2)
    return x - x.mean(axis=axes, keepdims=True)


def _remove_mean_inplace(arr, mode, chunk=512):
    mode = _check_mr(mode)
    if mode == "none":
        return arr
    for a in range(0, arr.shape[0], chunk):
        arr[a:a + chunk] = remove_mean(arr[a:a + chunk], mode)
    return arr


def get_stats_global(signal_paths, noise_paths, sim_ids, noise_scale, mean_remove="none"):
    """Single global (mean, std) across ALL redshifts combined."""
    all_signal_vals = []
    all_noise_var = 0.0
    n_noise_files = 0
    for s_path, n_path in zip(signal_paths, noise_paths):
        signals = np.fromfile(s_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        noises = np.fromfile(n_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        signals = signals[:9827][sim_ids]

        print(signals[200].shape)
        _remove_mean_inplace(signals, mean_remove)
        _remove_mean_inplace(noises, mean_remove)
        all_signal_vals.append(signals.ravel())
        all_noise_var += noises.var()
        n_noise_files += 1

    combined = np.concatenate(all_signal_vals)
    global_mean = np.float32(combined.mean())
    global_std = np.float32(np.sqrt(combined.var() + (noise_scale**2) * (all_noise_var / n_noise_files) + 1e-12))
    return [global_mean] * 3, [global_std] * 3


def _cube_var_split(X):
    cm = X.reshape(len(X), -1).mean(1)
    total = float(X.var())
    between = float(cm.var())
    return total, between, max(total - between, 0.0), cm


def get_stats(signal_paths, noise_paths, sim_ids, noise_scale,
              mean_remove="none", verbose=True):
    """Per-redshift (mean, std) of what the network actually sees."""
    mean_remove = _check_mr(mean_remove)
    means, stds = [], []

    for zi, (s_path, n_path) in enumerate(zip(signal_paths, noise_paths)):
        signals = np.fromfile(s_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        noises  = np.fromfile(n_path, dtype=np.float64).astype(np.float32).reshape(-1, 32, 32, 32)
        signals = signals[:9827][sim_ids]

        s_tot, s_btw, s_wit, s_cm = _cube_var_split(signals)
        n_tot, n_btw, n_wit, n_cm = _cube_var_split(noises)

        if mean_remove == "cube":
            s_var, n_var = s_wit, n_wit
            s_mean = 0.0
        elif mean_remove == "slice":
            sl = signals.reshape(len(signals), 32, -1)
            s_var = float((sl - sl.mean(2, keepdims=True)).var())
            nl = noises.reshape(len(noises), 32, -1)
            n_var = float((nl - nl.mean(2, keepdims=True)).var())
            s_mean = 0.0
        else:
            s_var, n_var, s_mean = s_tot, n_tot, float(signals.mean())

        n_mean = float(noises.mean()) if mean_remove == "none" else 0.0
        mean = np.float32(s_mean + noise_scale * n_mean)
        std  = np.float32(np.sqrt(s_var + (noise_scale ** 2) * n_var + 1e-12))
        means.append(mean); stds.append(std)

        if verbose:
            z = os.path.basename(s_path).split("z=")[-1].replace(".dat", "")
            print(f"\n--- z={z}  ({len(signals)} sims, {len(noises)} noise cubes) ---")
            print(f"  SIGNAL  total var {s_tot:11.4e}   mean {signals.mean():+.4e}")
            print(f"          in cube mean {s_btw:11.4e}  = {100*s_btw/s_tot:5.1f}% of total"
                  f"   <- destroyed by mean_remove='cube'")
            print(f"          fluctuations {s_wit:11.4e}  = {100*s_wit/s_tot:5.1f}%")
            print(f"          cube means: std {s_cm.std():.4e}  "
                  f"range [{s_cm.min():+.4e}, {s_cm.max():+.4e}]")
            print(f"  NOISE   total var {n_tot:11.4e}   mean {noises.mean():+.4e}")
            print(f"          in cube mean {n_btw:11.4e}  = {100*n_btw/n_tot:5.2f}% of total")
            print(f"          N_eff = within/between = {n_wit/max(n_btw,1e-30):,.0f}"
                  f"   (32768 if voxels were independent)")
            print(f"          std bias if noise mean NOT removed: "
                  f"{100*(np.sqrt(n_tot/max(n_wit,1e-30))-1):+.2f}%")
            snr = np.sqrt(s_wit / max((noise_scale**2) * n_wit, 1e-30))
            print(f"  SNR (fluctuation std, after mean removal) = {snr:7.3f}"
                  f"   at noise_scale={noise_scale}")
            print(f"  USED    mean {mean:+.4e}   std {std:.4e}   "
                  f"[mean_remove={mean_remove}]")

    return means, stds


def _lattice_flip(cube, axis):
    return torch.roll(torch.flip(cube, dims=[axis]), 1, dims=axis)


def _lattice_rot90(cube, k, dims):
    a, b = dims
    for _ in range(int(k) % 4):
        cube = _lattice_flip(cube.transpose(a, b), a)
    return cube


_AUG_GROUPS = ("full", "data")


def _check_aug(group):
    group = "full" if group in (None, True) else str(group).lower()
    if group not in _AUG_GROUPS:
        raise ValueError(f"augment_group must be one of {_AUG_GROUPS}, got {group!r}")
    return group


def augment_cube(cube, is_6d=False, group="full"):
    """Random element of O_h (order 48) composed with a periodic translation."""
    spatial_axes = [3, 4, 5] if is_6d else [1, 2, 3]
    rot_planes = ([3, 4], [3, 5], [4, 5]) if is_6d else ([1, 2], [1, 3], [2, 3])

    if _check_aug(group) == "data":
        d, h, w = spatial_axes
        if np.random.random() > 0.5:
            cube = _lattice_flip(cube, d)
        if np.random.random() > 0.5:
            cube = _lattice_flip(_lattice_flip(cube, h), w)
        shifts = tuple(int(np.random.randint(0, 32)) for _ in range(3))
        return torch.roll(cube, shifts=shifts, dims=spatial_axes)

    for axis in spatial_axes:
        if np.random.random() > 0.5:
            cube = _lattice_flip(cube, axis)

    for dims in rot_planes:
        cube = _lattice_rot90(cube, np.random.randint(0, 4), dims)

    shifts = tuple(int(np.random.randint(0, 32)) for _ in range(3))
    return torch.roll(cube, shifts=shifts, dims=spatial_axes)


class CubeDataset(Dataset):
    def __init__(self, s_paths, n_paths, params_path, sim_ids_path, means, stds,
                 augment, noise_scale, norm_flag, model_choice, mean_remove="none",
                 aux_mean=False, augment_group="full"):
        self.mean_remove = _check_mr(mean_remove)
        self.aux_mean = bool(aux_mean)
        self.params = np.load(params_path).astype(np.float32)

        self.params_norm = self.params.copy()
        self.params_norm[:, 0] = (np.log10(self.params_norm[:, 0] / 0.00192) - (-1.)) / (1. - (-1.))
        self.params_norm[:, 1] = (np.log10(self.params_norm[:, 1]) - np.log10(738.9)) / (np.log10(10504.7) - np.log10(738.9))
        self.params_norm[:, 3] = (self.params_norm[:, 3] - 8.) / (9.6 - 8.)
        self.params_norm[:, 4] = (self.params_norm[:, 4] - 0.05) / (0.5 - 0.05)

        self.sim_ids = np.load(sim_ids_path).astype(int)
        assert len(self.params) == len(self.sim_ids)

        self.signals = [np.fromfile(p, dtype=np.float64).astype(np.float32)
                        .reshape(-1, 32, 32, 32)[:9827][self.sim_ids] for p in s_paths]


        self.noises = [np.fromfile(p, dtype=np.float64).astype(np.float32)
                       .reshape(-1, 32, 32, 32) for p in n_paths]

        self.mu_std = None
        if self.aux_mean:
            cm = np.stack([sig.reshape(len(sig), -1).mean(1) for sig in self.signals], 1)
            self.mu_std = ((cm - cm.mean(0)) / (cm.std(0) + 1e-12)).astype(np.float32)

        self.means = means
        self.stds = stds
        self.augment = augment
        self.augment_group = _check_aug(augment_group)
        self.noise_scale = noise_scale
        self.norm_flag = bool(norm_flag)
        self.is_6d = (model_choice.lower() == "conv4d")

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        original_sim_id = self.sim_ids[idx]
        stacked_cube = []
        for i in range(3):
            noise_idx = np.random.randint(0, self.noises[i].shape[0])
            noisy_cube = self.signals[i][idx] + self.noise_scale * self.noises[i][noise_idx]
            noisy_cube = remove_mean(noisy_cube, self.mean_remove)
            if self.norm_flag:
                norm_cube = (noisy_cube - self.means[i]) / (self.stds[i] + 1e-8)
            else:
                norm_cube = noisy_cube
            stacked_cube.append(norm_cube)

        data = torch.from_numpy(np.stack(stacked_cube).astype(np.float32))


        if self.is_6d:
            data = data.unsqueeze(0)
            if self.augment:
                data = augment_cube(data.unsqueeze(0), is_6d=True,
                                    group=self.augment_group).squeeze(0)
        elif self.augment:
            data = augment_cube(data, is_6d=False, group=self.augment_group)

        target = self.params_norm[idx, [0, 1, 2, 3]]
        if self.mu_std is not None:
            target = np.concatenate([target, self.mu_std[idx]])
        target = torch.from_numpy(target.astype(np.float32).copy())
        return data, target, int(original_sim_id)


def prepare_cube_loaders(cfg_data, c, seed, val_frac):
    sim_ids = np.load(cfg_data["sim_ids_path"]).astype(int)

    model_choice = c.get("model", "seblock").lower()

    mean_remove = _check_mr(c.get("mean_remove", "none"))
    print(f"--> mean_remove = {mean_remove}", flush=True)
    augment_group = _check_aug(c.get("augment_group", "full"))
    print(f"--> augment = {c['augment']}  augment_group = {augment_group}", flush=True)

    if model_choice == "conv4d":
        means, stds = get_stats_global(cfg_data["s_paths"], cfg_data["n_paths"], sim_ids, c["noise_scale"], mean_remove)
    else:
        means, stds = get_stats(cfg_data["s_paths"], cfg_data["n_paths"], sim_ids, c["noise_scale"], mean_remove)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(sim_ids))
    rng.shuffle(all_indices)
    n_val = int(val_frac * len(all_indices))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    norm_flag = c.get("init_norm", True)
    _mh = (c.get("arch") or {}).get("mean_head", False)
    aux_mean = _mh not in (None, False, "none", "false")

    full_train_dataset = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"],
                                     cfg_data["sim_ids_path"], means, stds, augment=c["augment"],
                                     noise_scale=c["noise_scale"], norm_flag=norm_flag, model_choice=model_choice,
                                     mean_remove=mean_remove, aux_mean=aux_mean,
                                     augment_group=augment_group)
    full_val_dataset = CubeDataset(cfg_data["s_paths"], cfg_data["n_paths"], cfg_data["params_path"],
                                   cfg_data["sim_ids_path"], means, stds, augment=False,
                                   noise_scale=c["noise_scale"], norm_flag=norm_flag, model_choice=model_choice,
                                     mean_remove=mean_remove, aux_mean=aux_mean)

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 1
    _cfg_nw = c.get("num_workers", None)
    num_workers = int(_cfg_nw) if _cfg_nw is not None else max(0, usable - 1)
    num_workers = min(num_workers, max(0, usable - 1))
    print(f"--> usable cores={usable}  num_workers={num_workers}", flush=True)


    train_loader = DataLoader(train_dataset, batch_size=c["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader, full_val_dataset


class DeterministicNoisyExport(Dataset):
    """Deterministic NLE summary generation table; exports PRISTINE theta."""
    def __init__(self, base_dataset, n_noise_per_sim, noise_scale, noise_start=0, noise_end=None):
        if isinstance(base_dataset, torch.utils.data.Subset):
            self.base = base_dataset.dataset
        else:
            self.base = base_dataset

        self.total_n_noise_per_sim = n_noise_per_sim
        self.noise_start = noise_start
        self.noise_end = noise_end if noise_end is not None else n_noise_per_sim
        assert 0 <= self.noise_start < self.noise_end <= n_noise_per_sim
        self.n_noise_this_part = self.noise_end - self.noise_start
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.base) * self.n_noise_this_part

    def __getitem__(self, global_idx):
        sim_idx = global_idx // self.n_noise_this_part
        noise_rep = self.noise_start + (global_idx % self.n_noise_this_part)
        original_sim_id = int(self.base.sim_ids[sim_idx])

        stacked_cube, noise_ids = [], []
        for ch in range(3):
            n_noise_total = self.base.noises[ch].shape[0]
            noise_idx = (original_sim_id * self.total_n_noise_per_sim + noise_rep + 10007 * ch) % n_noise_total
            noisy_cube = self.base.signals[ch][sim_idx] + self.noise_scale * self.base.noises[ch][noise_idx]
            noisy_cube = remove_mean(noisy_cube, self.base.mean_remove)
            if self.base.norm_flag:
                norm_cube = (noisy_cube - self.base.means[ch]) / (self.base.stds[ch] + 1e-8)
            else:
                norm_cube = noisy_cube
            stacked_cube.append(norm_cube.astype(np.float32))
            noise_ids.append(noise_idx)

        data = torch.from_numpy(np.stack(stacked_cube, axis=0).astype(np.float32))
        if self.base.is_6d:
            data = data.unsqueeze(0)

        target = torch.from_numpy(self.base.params_norm[sim_idx, :4].copy().astype(np.float32))
        return data, target, original_sim_id, torch.tensor(noise_ids, dtype=torch.long)
