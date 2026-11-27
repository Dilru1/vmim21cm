"""Architecture-justification ablations of seblock_amp (compressor.model: cnn_ablate)."""
import torch.nn as nn
import torch.nn.functional as F

from .cnn_amp import AmpResNet3DCompressor, _ARCH_KEYS as _AMP_KEYS
from .cnn_seblock_up import ResidualBlock3d


class PlainBlock3d(ResidualBlock3d):
    """ResidualBlock3d without the skip: conv-GN-lrelu-conv-GN-SE-lrelu."""

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.leaky_relu(out, 0.1)


class AblationCompressor(AmpResNet3DCompressor):
    def __init__(self, *args, residual=True, **kw):
        super().__init__(*args, **kw)
        self.residual = bool(residual)
        if not self.residual:
            for blk in self._stage_blocks:
                blk.__class__ = PlainBlock3d
                del blk.shortcut


_ARCH_KEYS = tuple(_AMP_KEYS) + ("residual",)


def build_ablate_cnn_from_cfg(c, t_dim, n_params, direct=False):
    """compressor.arch -> AblationCompressor."""
    arch = dict(c.get("arch", {}) or {})
    unknown = set(arch) - set(_ARCH_KEYS)
    if unknown:
        raise ValueError(f"unknown compressor.arch keys for cnn_ablate: {sorted(unknown)}")
    use_z = c.get("use_redshifts")
    if use_z is not None:
        arch.setdefault("in_channels", len(use_z))
    return AblationCompressor(t_dim=t_dim, n_params=n_params, direct=direct, **arch)


from .cnn_seblock_up import ResNet3DCompressor, _ARCH_KEYS as _SEBLOCK_KEYS


class SeblockAblationCompressor(ResNet3DCompressor):
    def __init__(self, *args, residual=True, **kw):
        super().__init__(*args, **kw)
        self.residual = bool(residual)
        if not self.residual:
            for blk in self._stage_blocks:
                blk.__class__ = PlainBlock3d
                del blk.shortcut


_SEBLOCK_ABLATE_KEYS = tuple(_SEBLOCK_KEYS) + ("residual",)


def build_seblock_ablate_from_cfg(c, t_dim, n_params, direct=False):
    """compressor.arch -> SeblockAblationCompressor."""
    arch = dict(c.get("arch", {}) or {})
    unknown = set(arch) - set(_SEBLOCK_ABLATE_KEYS)
    if unknown:
        raise ValueError(f"unknown compressor.arch keys for seblock_ablate: {sorted(unknown)}")
    use_z = c.get("use_redshifts")
    if use_z is not None:
        arch.setdefault("in_channels", len(use_z))
    return SeblockAblationCompressor(t_dim=t_dim, n_params=n_params, direct=direct, **arch)
