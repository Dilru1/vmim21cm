"""Extra readouts for the 3D CNN compressor."""
import math

import torch
import torch.nn as nn


def _kgrid(n):
    k = torch.fft.fftfreq(n) * n
    kz = torch.fft.rfftfreq(n) * n
    k0 = k[:, None, None].expand(n, n, kz.numel())
    k1 = k[None, :, None].expand(n, n, kz.numel())
    k2 = kz[None, None, :].expand(n, n, kz.numel())
    kmag = torch.sqrt(k0 ** 2 + k1 ** 2 + k2 ** 2)
    w = torch.full((kz.numel(),), 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    w = w[None, None, :].expand_as(kmag)
    return kmag, (k0, k1, k2), w


class SpectralReadout(nn.Module):
    """Differentiable P(k), cross-redshift r_ij(k) and one-point moments."""

    def __init__(self, n=32, n_ch=3, nbins=8, cross=True, moments=True, mu_bins=1):
        super().__init__()
        self.n, self.n_ch = int(n), int(n_ch)
        self.cross, self.moments = bool(cross), bool(moments)
        kmag, (k0, _, _), w = _kgrid(self.n)

        edges = torch.logspace(math.log10(0.999), math.log10(1.001 * self.n / 2), int(nbins) + 1)
        kb = torch.bucketize(kmag, edges, right=True) - 1
        valid = (kb >= 0) & (kb < int(nbins))
        if int(mu_bins) == 2:
            mu = (k0.abs() / kmag.clamp_min(1e-6))
            kb = kb * 2 + (mu >= 0.5).long()
            nb = 2 * int(nbins)
        elif int(mu_bins) == 1:
            nb = int(nbins)
        else:
            raise ValueError("mu_bins must be 1 or 2")

        idx = torch.nonzero(valid.flatten(), as_tuple=False).squeeze(1)
        M = torch.zeros(idx.numel(), nb)
        M[torch.arange(idx.numel()), kb.flatten()[idx]] = w.flatten()[idx]
        counts = M.sum(0)
        keep = counts > 0
        M = M[:, keep] / counts[keep]
        self.register_buffer("idx", idx)
        self.register_buffer("M", M)
        self.nb = int(keep.sum())
        self.pairs = [(i, j) for i in range(self.n_ch) for j in range(i + 1, self.n_ch)]

    @property
    def out_dim(self):
        d = self.n_ch * self.nb
        if self.cross:
            d += len(self.pairs) * self.nb
        if self.moments:
            d += 3 * self.n_ch
        return d

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B = x.shape[0]
            xc = x - x.mean(dim=(2, 3, 4), keepdim=True)
            Fk = torch.fft.rfftn(xc, dim=(2, 3, 4)).flatten(2)[..., self.idx]
            P = (Fk.real ** 2 + Fk.imag ** 2) @ self.M
            feats = [torch.log(P + 1e-8).flatten(1)]
            if self.cross and self.pairs:
                rs = []
                for i, j in self.pairs:
                    Pij = (Fk[:, i].real * Fk[:, j].real + Fk[:, i].imag * Fk[:, j].imag) @ self.M
                    rs.append(Pij / torch.sqrt(P[:, i] * P[:, j] + 1e-12))
                feats.append(torch.stack(rs, 1).flatten(1))
            if self.moments:
                flat = xc.flatten(2)
                sd = flat.std(dim=2) + 1e-8
                z = flat / sd.unsqueeze(2)
                skew = (z ** 3).mean(2)
                kurt = (z ** 4).mean(2) - 3.0
                feats.append(torch.cat([torch.log(sd), skew, kurt], 1))
            return torch.cat(feats, 1).reshape(B, -1)


class CovPool(nn.Module):
    """Second-order pooling: sqrt of the channel covariance of a feature map."""

    def __init__(self, c_in, c_red=32, ns_iter=10, eps=1e-4):
        super().__init__()
        self.reduce = nn.Conv3d(c_in, c_red, kernel_size=1, bias=False)
        self.c, self.ns_iter, self.eps = int(c_red), int(ns_iter), float(eps)
        iu = torch.triu_indices(self.c, self.c)
        self.register_buffer("iu0", iu[0])
        self.register_buffer("iu1", iu[1])

    @property
    def out_dim(self):
        return self.c * (self.c + 1) // 2

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.reduce(x.float()).flatten(2)
            z = z - z.mean(dim=2, keepdim=True)
            C = z @ z.transpose(1, 2) / (z.shape[2] - 1)
            I = torch.eye(self.c, device=C.device, dtype=C.dtype).expand_as(C)
            C = C + self.eps * I
            tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8)
            A = C / tr[:, None, None]
            Y, Z = A, I
            for _ in range(self.ns_iter):
                T = 0.5 * (3.0 * I - Z @ Y)
                Y, Z = Y @ T, T @ Z
            S = Y * torch.sqrt(tr)[:, None, None]
            return S[:, self.iu0, self.iu1]


class FeatureStandardizer(nn.Module):
    """DATASET-level standardisation: (f - running_mean) / running_std."""

    def __init__(self, dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.momentum, self.eps = float(momentum), float(eps)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))

    def forward(self, f):
        f = f.float()
        if self.training and f.shape[0] > 1:
            with torch.no_grad():
                mom = max(self.momentum, 1.0 / float(self.n_seen.item() + 1))
                self.mean.lerp_(f.mean(0), mom)
                self.var.lerp_(f.var(0, unbiased=False), mom)
                self.n_seen += 1
        return (f - self.mean) / torch.sqrt(self.var + self.eps)


class OnePointReadout(nn.Module):
    """Multi-scale one-point PDF of the INPUT cube, per redshift channel."""

    def __init__(self, n=32, n_ch=3, smooth=(0.0, 1.0, 2.0),
                 quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), cross=True):
        super().__init__()
        self.n, self.n_ch, self.cross = int(n), int(n_ch), bool(cross)
        self.smooth = [float(r) for r in smooth]
        kmag, _, _ = _kgrid(self.n)
        k2 = (2 * math.pi * kmag / self.n) ** 2
        self.register_buffer("kern", torch.stack([torch.exp(-0.5 * k2 * r * r) for r in self.smooth]))
        self.register_buffer("q", torch.tensor([float(v) for v in quantiles]))
        self.pairs = [(i, j) for i in range(self.n_ch) for j in range(i + 1, self.n_ch)]

    @property
    def out_dim(self):
        per = self.n_ch * (3 + self.q.numel()) + (len(self.pairs) if self.cross else 0)
        return len(self.smooth) * per

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            sp = x.shape[2:]
            Fx = torch.fft.rfftn(x, dim=(2, 3, 4)) if any(self.smooth) else None
            feats = []
            for r, kern in zip(self.smooth, self.kern):
                xs = x if r == 0 else torch.fft.irfftn(Fx * kern, s=sp, dim=(2, 3, 4))
                flat = xs.flatten(2)
                flat = flat - flat.mean(2, keepdim=True)
                sd = flat.std(2) + 1e-8
                z = flat / sd.unsqueeze(2)
                feats += [torch.log(sd), (z ** 3).mean(2), (z ** 4).mean(2) - 3.0]
                if self.q.numel():
                    qz = torch.quantile(z, self.q, dim=2)
                    feats.append(qz.permute(1, 2, 0).flatten(1))
                if self.cross and self.pairs:
                    feats.append(torch.stack([(z[:, i] * z[:, j]).mean(1)
                                              for i, j in self.pairs], 1))
            return torch.cat(feats, 1)
