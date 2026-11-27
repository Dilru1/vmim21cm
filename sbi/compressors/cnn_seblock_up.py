import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .readouts import SpectralReadout, CovPool


DEFAULT_CHANNELS = [32, 32, 64, 128, 256, 512]
DEFAULT_STRIDES  = [1, 1, 2, 2, 2]

class OhConv3d(nn.Module):
    """3x3x3 conv with weights tied across the O_h orbits of the kernel offsets."""
    ORBITS = [[(0, 0, 0)],
              [p for p in itertools.product((-1, 0, 1), repeat=3) if sum(map(abs, p)) == 1],
              [p for p in itertools.product((-1, 0, 1), repeat=3) if sum(map(abs, p)) == 2],
              [p for p in itertools.product((-1, 0, 1), repeat=3) if sum(map(abs, p)) == 3]]

    def __init__(self, ci, co, stride=1, padding_mode="circular"):
        super().__init__()
        if padding_mode == "zeros":
            padding_mode = "constant"
        self.ci, self.co, self.stride, self.pm = ci, co, stride, padding_mode
        self.w = nn.Parameter(torch.randn(co, ci, 4) * (2.0 / (ci * 27)) ** 0.5)
        idx = torch.zeros(4, 3, 3, 3)
        for k, orb in enumerate(self.ORBITS):
            for (a, b, c) in orb:
                idx[k, a + 1, b + 1, c + 1] = 1.0
        self.register_buffer("masks", idx)

    def forward(self, x):
        W = torch.einsum("oik,kabc->oiabc", self.w, self.masks)
        x = F.pad(x, (1,) * 6, mode=self.pm)
        return F.conv3d(x, W, stride=self.stride)


def _gn_groups(channels, target):
    g = min(int(target), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1, 1).expand_as(x)


class ResidualBlock3d(nn.Module):
    """3D residual block with optional downsampling and SE block."""
    def __init__(self, ci, co, stride=1, reduction=8, padding_mode="zeros",
                 use_se=True, equivariant=False):
        super().__init__()
        def _conv3(a, b, st):
            if equivariant:
                return OhConv3d(a, b, stride=st, padding_mode=padding_mode)
            return nn.Conv3d(a, b, kernel_size=3, stride=st, padding=1,
                             bias=False, padding_mode=padding_mode)
        self.conv1 = _conv3(ci, co, stride)
        self.bn1 = nn.GroupNorm(min(8, co // 4), co)
        self.conv2 = _conv3(co, co, 1)
        self.bn2 = nn.GroupNorm(min(8, co // 4), co)
        self.se = SEBlock3d(co, reduction=reduction) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, co // 4), co)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1)


def _pool_mean_std(x):
    flat = x.flatten(2)
    return torch.cat([flat.mean(dim=2), flat.std(dim=2)], dim=1)

def _pool_stats(x, stats=("mean", "std")):
    flat = x.flatten(2)
    out = []
    for s in stats:
        out.append(flat.mean(dim=2) if s == "mean" else flat.std(dim=2))
    return torch.cat(out, dim=1)


class ResNet3DCompressor(nn.Module):
    """3D residual multi-scale CNN compressor."""
    def __init__(self, t_dim=8, n_params=4, direct=False,
                 channels=None, strides=None, readout_from=None,
                 width_mult=1.0, se_reduction=8, use_se=True,
                 pool_stats=("mean", "std"),
                 in_channels=3, circular_padding=False, equivariant=False,
                 spectral_readout=None, cov_readout=None, cov_dim=32):
        super().__init__()
        pad_mode = "circular" if circular_padding else "zeros"
        if isinstance(equivariant, str):
            if equivariant.lower() not in ("oh", "none", "false", ""):
                raise ValueError(f"equivariant must be false or 'Oh', got {equivariant!r}")
            equivariant = equivariant.lower() == "oh"
        self.equivariant = bool(equivariant)
        if self.equivariant and not circular_padding:
            print("--> WARNING: equivariant=Oh with circular_padding=false. "
                  "Zero padding breaks equivariance at the box faces; the "
                  "boxes are periodic, so set circular_padding: true.", flush=True)
        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}")

        if channels is None:
            channels = [max(1, int(round(c * float(width_mult)))) for c in DEFAULT_CHANNELS]
        else:
            channels = [int(c) for c in channels]
        n_blocks = len(channels) - 1
        if n_blocks < 1:
            raise ValueError(f"channels must have >=2 entries (stem + >=1 block), got {channels}")

        if strides is None:
            strides = list(DEFAULT_STRIDES) if n_blocks == len(DEFAULT_STRIDES) \
                      else [1] + [2] * (n_blocks - 1)
        else:
            strides = [int(s) for s in strides]
        if len(strides) != n_blocks:
            raise ValueError(f"strides must have length n_blocks={n_blocks}, got {len(strides)}")

        if readout_from is None:
            readout_stages = list(range(2, n_blocks + 1))
        else:
            readout_stages = [int(s) for s in readout_from]
        for s in readout_stages:
            if not (1 <= s <= n_blocks):
                raise ValueError(f"readout_from stage {s} out of range 1..{n_blocks}")
        if not readout_stages:
            raise ValueError("readout_from resolved to an empty set of stages")
        self.readout_stages = readout_stages
        self.channels, self.strides = channels, strides

        self.pool_stats = tuple(pool_stats)
        bad = set(self.pool_stats) - {"mean", "std"}
        if bad or not self.pool_stats:
            raise ValueError(f"pool_stats must be a non-empty subset of "
                             f"('mean','std'), got {pool_stats}")

        c0 = channels[0]
        self.stem = nn.Sequential(
            (OhConv3d(int(in_channels), c0, stride=1, padding_mode=pad_mode)
             if self.equivariant else
             nn.Conv3d(int(in_channels), c0, kernel_size=3, stride=1, padding=1,
                       bias=False, padding_mode=pad_mode)),
            nn.GroupNorm(_gn_groups(c0, 4), c0),
            nn.LeakyReLU(0.1)
        )

        self._stage_blocks = []
        prev = c0
        for i in range(n_blocks):
            blk = ResidualBlock3d(prev, channels[i + 1], stride=strides[i],
                                  reduction=se_reduction, padding_mode=pad_mode,
                                  use_se=use_se, equivariant=self.equivariant)
            setattr(self, f"layer{i + 1}", blk)
            self._stage_blocks.append(blk)
            prev = channels[i + 1]

        combined_features_dim = len(self.pool_stats) * sum(
            channels[s] for s in readout_stages)

        self.spectral = None
        if spectral_readout:
            kw = dict(spectral_readout) if isinstance(spectral_readout, dict) else {}
            kw.setdefault("n_ch", int(in_channels))
            self.spectral = SpectralReadout(**kw)
            self.spectral_norm = nn.LayerNorm(self.spectral.out_dim)
            combined_features_dim += self.spectral.out_dim

        self.cov_stages = [int(s) for s in (cov_readout or [])]
        for s in self.cov_stages:
            if not (1 <= s <= n_blocks):
                raise ValueError(f"cov_readout stage {s} out of range 1..{n_blocks}")
        self.covpools = None
        if self.cov_stages:
            self.covpools = nn.ModuleList([CovPool(channels[s], c_red=int(cov_dim))
                                           for s in self.cov_stages])
            cov_out = sum(cp.out_dim for cp in self.covpools)
            self.cov_norm = nn.LayerNorm(cov_out)
            combined_features_dim += cov_out

        self.fc_summary = nn.Sequential(
            nn.Linear(combined_features_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(128, t_dim)
        )

        self.fc_aux = None if self.direct else nn.Sequential(
            nn.Linear(combined_features_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, n_params)
        )

    def forward(self, x):
        extra = []
        if self.spectral is not None:
            extra.append(self.spectral_norm(self.spectral(x)))
        x = self.stem(x)
        outs = []
        for blk in self._stage_blocks:
            x = blk(x)
            outs.append(x)

        f = torch.cat([_pool_stats(outs[s - 1], self.pool_stats)
                       for s in self.readout_stages], dim=1)
        if self.covpools is not None:
            cov = torch.cat([cp(outs[s - 1]) for cp, s in zip(self.covpools, self.cov_stages)], dim=1)
            extra.append(self.cov_norm(cov))
        if extra:
            f = torch.cat([f] + [e.to(f.dtype) for e in extra], dim=1)

        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)


_ARCH_KEYS = ("channels", "strides", "readout_from", "width_mult", "se_reduction",
              "use_se", "pool_stats",
              "in_channels", "circular_padding", "equivariant",
              "spectral_readout", "cov_readout", "cov_dim")


def build_cnn_from_cfg(c, t_dim, n_params, direct=False):
    """Build the compressor from the yaml `compressor.arch` block."""
    arch = dict(c.get("arch", {}) or {})
    unknown = set(arch) - set(_ARCH_KEYS)
    if unknown:
        raise ValueError(f"unknown compressor.arch keys: {sorted(unknown)}")
    use_z = c.get("use_redshifts")
    if use_z is not None:
        arch.setdefault("in_channels", len(use_z))
    return ResNet3DCompressor(t_dim=t_dim, n_params=n_params, direct=direct, **arch)
