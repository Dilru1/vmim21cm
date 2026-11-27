"""Standalone 3D-CNN compressor: load trained weights and compress observed cubes to summaries t."""
import json
from pathlib import Path

import numpy as np
import torch

from .compressors import CNNCompressor
from .cubes import remove_cube_mean


def _shape_from_state(state):
    """(in_channels, t_dim, n_params, direct) read off a CNNCompressor state dict."""
    in_ch = state["stem.0.weight"].shape[1]
    t_dim = state["fc_summary.8.weight"].shape[0]
    direct = "fc_aux.8.weight" not in state
    n_params = t_dim if direct else state["fc_aux.8.weight"].shape[0]
    return int(in_ch), int(t_dim), int(n_params), direct


class CubeCompressor:
    """Trained CNNCompressor plus its per-channel standardisation; call compress(cubes) -> t."""

    def __init__(self, model, means, stds, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.means = np.asarray(means, np.float32)
        self.stds = np.asarray(stds, np.float32)

    @classmethod
    def from_checkpoint(cls, weights, means, stds, device="cpu"):
        """Build from a compressor.pt state dict and channel means/stds."""
        state = torch.load(weights, map_location="cpu")
        in_ch, t_dim, n_params, direct = _shape_from_state(state)
        model = CNNCompressor(in_ch, t_dim, n_params, direct)
        model.load_state_dict(state)
        return cls(model, means, stds, device)

    @classmethod
    def from_run(cls, run_dir, device="cpu"):
        """Build from a stage-1 run directory ({out_root}/{name}/c{seed} or its compressor/)."""
        d = Path(run_dir)
        d = d / "compressor" if (d / "compressor").is_dir() else d
        stats = d / "cube_stats.npz"
        if not stats.exists():
            raise FileNotFoundError(f"{stats} missing: pass means/stds to from_checkpoint instead")
        s = np.load(stats)
        return cls.from_checkpoint(d / "compressor.pt", s["means"], s["stds"], device)

    @property
    def t_dim(self):
        return self.model.fc_summary[-1].out_features

    def preprocess(self, cubes):
        """(N, C, box, box, box) signal + noise -> mean-removed, standardised float32 tensor."""
        x = remove_cube_mean(np.asarray(cubes, np.float32))
        x = (x - self.means[None, :, None, None, None]) / (self.stds[None, :, None, None, None] + 1e-8)
        return torch.from_numpy(x.astype(np.float32))

    @torch.no_grad()
    def compress(self, cubes, batch_size=64):
        """(N, C, box, box, box) or (C, box, box, box) cubes -> t of shape (N, t_dim) or (t_dim,)."""
        cubes = np.asarray(cubes)
        single = cubes.ndim == 4
        if single:
            cubes = cubes[None]
        out = []
        for s in range(0, len(cubes), batch_size):
            x = self.preprocess(cubes[s:s + batch_size]).to(self.device)
            out.append(self.model(x)[0].float().cpu().numpy())
        t = np.concatenate(out)
        return t[0] if single else t

    __call__ = compress

    def info(self):
        """Architecture and standardisation summary."""
        return json.dumps({"in_channels": self.model.stem[0].in_channels, "t_dim": self.t_dim,
                           "direct": self.model.direct, "means": self.means.tolist(),
                           "stds": self.stds.tolist()})
