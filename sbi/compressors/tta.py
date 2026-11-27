"""Symmetrised summaries for export."""
import torch
import torch.nn as nn


class SymmetrizedCompressor(nn.Module):
    def __init__(self, base, n_group=8):
        super().__init__()
        if n_group not in (1, 2, 4, 8, 16):
            raise ValueError("n_group must be 1, 2, 4, 8 or 16")
        self.base = base
        els = [(k, f, l) for l in (False, True) for f in (False, True) for k in range(4)]
        self.elements = els[:n_group]

    @staticmethod
    def _apply(x, k, flip_t, flip_los):
        if flip_los:
            x = torch.flip(x, dims=[2])
        if flip_t:
            x = torch.flip(x, dims=[3])
        if k:
            x = torch.rot90(x, k, dims=[3, 4])
        return x

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError("SymmetrizedCompressor expects (B, C, D, H, W) cubes")
        t_sum, a_sum, has_aux = None, None, False
        for k, ft, fl in self.elements:
            out = self.base(self._apply(x, k, ft, fl))
            t = out[0] if isinstance(out, tuple) else out
            t_sum = t.float() if t_sum is None else t_sum + t.float()
            if isinstance(out, tuple):
                has_aux = True
                a = out[1].float()
                a_sum = a if a_sum is None else a_sum + a
        n = float(len(self.elements))
        t_mean = t_sum / n
        return (t_mean, a_sum / n) if has_aux else t_mean
