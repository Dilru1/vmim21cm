"""Amplitude-aware 3D residual compressor (compressor.model: seblock_amp)."""
import torch
import torch.nn as nn

from .cnn_seblock_up import (DEFAULT_CHANNELS, DEFAULT_STRIDES, ResidualBlock3d,
                             _gn_groups, _pool_stats)
from .readouts import SpectralReadout


class FrozenStandardize(nn.Module):
    """(x - mu) / sd per feature."""

    def __init__(self, dim, n_batches=64, eps=1e-6):
        super().__init__()
        self.n_batches, self.eps = int(n_batches), float(eps)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.register_buffer("s1", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("s2", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("n", torch.zeros((), dtype=torch.float64))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    @property
    def frozen(self):
        return int(self.seen) >= self.n_batches

    def forward(self, x):
        if self.training and not self.frozen:
            with torch.no_grad():
                xd = x.detach().double()
                self.s1 += xd.sum(0)
                self.s2 += (xd * xd).sum(0)
                self.n += xd.shape[0]
                self.seen += 1
                mu = self.s1 / self.n
                var = (self.s2 / self.n - mu * mu).clamp_min(0.0)
                self.mu.copy_(mu.to(self.mu.dtype))
                self.sd.copy_(var.sqrt().clamp_min(self.eps).to(self.sd.dtype))
        return (x - self.mu) / self.sd


def _norm1d(kind, dim, n_batches=64):
    if kind == "frozen":
        return FrozenStandardize(dim, n_batches=n_batches)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def _choice(name, val, allowed):
    val = "none" if val in (None, False) else str(val).lower()
    if val not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
    return val


class AmpResNet3DCompressor(nn.Module):
    def __init__(self, t_dim=8, n_params=4, direct=False,
                 channels=None, strides=None, readout_from=None,
                 width_mult=1.0, se_reduction=8, use_se=True,
                 pool_stats=("mean", "std"), in_channels=3, circular_padding=False,
                 stem_norm="group", fc_norm="layer",
                 amp_readout=None, amp_norm="frozen", amp_init_batches=64):
        super().__init__()
        self.stem_norm = _choice("stem_norm", stem_norm, ("group", "none"))
        self.fc_norm = _choice("fc_norm", fc_norm, ("layer", "none"))
        amp_norm = _choice("amp_norm", amp_norm, ("frozen", "none", "batch", "layer"))
        pad_mode = "circular" if circular_padding else "zeros"

        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(f"direct regression needs t_dim==n_params, got {t_dim}, {n_params}")

        if channels is None:
            channels = [max(1, int(round(c * float(width_mult)))) for c in DEFAULT_CHANNELS]
        else:
            channels = [int(c) for c in channels]
        n_blocks = len(channels) - 1
        if n_blocks < 1:
            raise ValueError(f"channels must have >=2 entries, got {channels}")
        if strides is None:
            strides = list(DEFAULT_STRIDES) if n_blocks == len(DEFAULT_STRIDES) \
                      else [1] + [2] * (n_blocks - 1)
        else:
            strides = [int(s) for s in strides]
        if len(strides) != n_blocks:
            raise ValueError(f"strides must have length {n_blocks}, got {len(strides)}")
        readout_stages = (list(range(2, n_blocks + 1)) if readout_from is None
                          else [int(s) for s in readout_from])
        if not readout_stages or any(not (1 <= s <= n_blocks) for s in readout_stages):
            raise ValueError(f"readout_from must be a non-empty subset of 1..{n_blocks}")
        self.readout_stages, self.channels, self.strides = readout_stages, channels, strides
        self.pool_stats = tuple(pool_stats)
        if not self.pool_stats or set(self.pool_stats) - {"mean", "std"}:
            raise ValueError(f"pool_stats must be a subset of ('mean','std'), got {pool_stats}")

        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv3d(int(in_channels), c0, kernel_size=3, stride=1, padding=1,
                      bias=False, padding_mode=pad_mode),
            nn.GroupNorm(_gn_groups(c0, 4), c0) if self.stem_norm == "group" else nn.Identity(),
            nn.LeakyReLU(0.1),
        )

        self._stage_blocks = []
        prev = c0
        for i in range(n_blocks):
            blk = ResidualBlock3d(prev, channels[i + 1], stride=strides[i],
                                  reduction=se_reduction, padding_mode=pad_mode,
                                  use_se=use_se, equivariant=False)
            setattr(self, f"layer{i + 1}", blk)
            self._stage_blocks.append(blk)
            prev = channels[i + 1]
        feat_dim = len(self.pool_stats) * sum(channels[s] for s in readout_stages)

        self.amp = None
        if amp_readout:
            kw = dict(amp_readout) if isinstance(amp_readout, dict) else {}
            kw.setdefault("n_ch", int(in_channels))
            self.amp = SpectralReadout(**kw)
            self.amp_norm = _norm1d(amp_norm, self.amp.out_dim, n_batches=amp_init_batches)
            feat_dim += self.amp.out_dim

        nf = lambda d: _norm1d(self.fc_norm, d)                       # noqa: E731
        self.fc_summary = nn.Sequential(
            nn.Linear(feat_dim, 256), nf(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(256, 128), nf(128), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(128, t_dim))
        self.fc_aux = None if self.direct else nn.Sequential(
            nn.Linear(feat_dim, 256), nf(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nf(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, n_params))

    def forward(self, x):
        amp = self.amp_norm(self.amp(x).to(x.dtype)) if self.amp is not None else None
        x = self.stem(x)
        outs = []
        for blk in self._stage_blocks:
            x = blk(x)
            outs.append(x)
        f = torch.cat([_pool_stats(outs[s - 1], self.pool_stats) for s in self.readout_stages], 1)
        if amp is not None:
            f = torch.cat([f, amp], 1)
        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)


_ARCH_KEYS = ("channels", "strides", "readout_from", "width_mult", "se_reduction",
              "use_se", "pool_stats", "in_channels", "circular_padding",
              "stem_norm", "fc_norm", "amp_readout", "amp_norm", "amp_init_batches")


def build_amp_cnn_from_cfg(c, t_dim, n_params, direct=False):
    """compressor.arch -> AmpResNet3DCompressor."""
    arch = dict(c.get("arch", {}) or {})
    unknown = set(arch) - set(_ARCH_KEYS)
    if unknown:
        raise ValueError(f"unknown compressor.arch keys for seblock_amp: {sorted(unknown)}")
    use_z = c.get("use_redshifts")
    if use_z is not None:
        arch.setdefault("in_channels", len(use_z))
    return AmpResNet3DCompressor(t_dim=t_dim, n_params=n_params, direct=direct, **arch)
