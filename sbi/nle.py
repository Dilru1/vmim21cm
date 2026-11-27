"""Stage-2 NLE density models: q(t | theta), pluggable families."""
import json
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG2PI = math.log(2.0 * math.pi)

class GMMConditional(nn.Module):
    """theta -> {weights, means, triangular factor} of a K-component MVN mixture."""

    def __init__(self, t_dim, n_params, n_mix=4, hidden=64,
                 param_style="softplus", diag_floor=1e-4,
                 legacy_branch=False, mean_style="linear"):
        super().__init__()
        self.t_dim, self.n_params, self.n_mix = t_dim, n_params, n_mix
        self.param_style = param_style
        self.diag_floor = float(diag_floor)
        self.legacy = (param_style == "semelin_legacy")
        self.mean_style = mean_style

        self.net = nn.Sequential(nn.Linear(n_params, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())

        def branch():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh()) if legacy_branch \
                else nn.Identity()

        self.b_w, self.b_mu, self.b_v, self.b_o = branch(), branch(), branch(), branch()

        n_off = t_dim * (t_dim - 1) // 2
        self.w = nn.Linear(hidden, n_mix)
        self.mu = nn.Linear(hidden, n_mix * t_dim)
        if self.legacy:
            self.v = nn.Linear(hidden, n_mix * t_dim)
            self.o = nn.Linear(hidden, n_mix * n_off)
        else:
            self.tr = nn.Linear(hidden, n_mix * (t_dim * (t_dim + 1) // 2))

        if self.legacy:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def _legacy_logprob(self, theta, t):
        h = self.net(theta)
        B, K, D = theta.shape[0], self.n_mix, self.t_dim

        logits = self.w(self.b_w(h))

        mu = self.mu(self.b_mu(h)).view(B, K, D)
        if self.mean_style == "sigmoid_scaled":
            mu = torch.sigmoid(mu) * 1.2 - 0.1

        U = torch.zeros(B, K, D, D, device=theta.device, dtype=theta.dtype)
        di = torch.arange(D, device=theta.device)
        U[:, :, di, di] = self.v(self.b_v(h)).view(B, K, D)
        if D > 1:
            iu = torch.triu_indices(D, D, offset=1, device=theta.device)
            U[:, :, iu[0], iu[1]] = self.o(self.b_o(h)).view(B, K, -1)

        d = t[:, None, :] - mu
        a = torch.einsum("bkij,bkj->bki", U, d)
        quad = -0.5 * (a ** 2).sum(-1)
        logdet = torch.log(torch.abs(U[:, :, di, di]) + 1e-37).sum(-1)

        comp = logdet + quad - 0.5 * D * _LOG2PI
        lse = torch.logsumexp((torch.log_softmax(logits, -1) + comp).double(), dim=-1)
        floor = torch.full_like(lse, math.log(1e-37))
        return torch.logaddexp(lse, floor).to(theta.dtype)

    def _mixture(self, theta):
        h = self.net(theta)
        B = theta.shape[0]
        logits = self.w(self.b_w(h))
        mu = self.mu(self.b_mu(h)).view(B, self.n_mix, self.t_dim)
        if self.mean_style == "sigmoid_scaled":
            mu = torch.sigmoid(mu) * 1.2 - 0.1
        L = torch.zeros(B, self.n_mix, self.t_dim, self.t_dim,
                        device=theta.device, dtype=theta.dtype)
        idx = torch.tril_indices(self.t_dim, self.t_dim, 0, device=theta.device)
        L[:, :, idx[0], idx[1]] = self.tr(h).view(B, self.n_mix, -1)
        di = torch.arange(self.t_dim, device=theta.device)
        if self.param_style == "softplus":
            L[:, :, di, di] = F.softplus(L[:, :, di, di]) + self.diag_floor
        else:
            L[:, :, di, di] = torch.exp(L[:, :, di, di]) + self.diag_floor
        return logits, mu, L

    def log_prob(self, theta, t):
        if self.legacy:
            return self._legacy_logprob(theta, t)
        logits, mu, L = self._mixture(theta)
        lp = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L) \
                  .log_prob(t[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, -1) + lp, dim=-1)


def _build_flow(kind, t_dim, n_params, hidden=64, n_layers=5,
                num_bins=8, tail_bound=5.0, residual=True, num_blocks=2):
    from nflows import transforms, distributions, flows
    from nflows.transforms.autoregressive import (
        MaskedAffineAutoregressiveTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )

    tl = []
    if kind == "made":
        tl.append(MaskedAffineAutoregressiveTransform(
            features=t_dim, hidden_features=hidden, context_features=n_params,
            num_blocks=2, use_residual_blocks=True, activation=F.relu))
    elif kind == "maf":
        for _ in range(n_layers):
            tl.append(MaskedAffineAutoregressiveTransform(
                features=t_dim, hidden_features=hidden, context_features=n_params,
                num_blocks=num_blocks, use_residual_blocks=residual, activation=F.relu))
            tl.append(transforms.RandomPermutation(features=t_dim))
    elif kind == "nsf":
        for _ in range(n_layers):
            tl.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=t_dim, hidden_features=hidden, context_features=n_params,
                num_blocks=num_blocks, num_bins=num_bins, tails="linear",
                tail_bound=tail_bound, use_residual_blocks=residual))
            tl.append(transforms.RandomPermutation(features=t_dim))
    else:
        raise ValueError(kind)
    return flows.Flow(transform=transforms.CompositeTransform(tl),
                      distribution=distributions.StandardNormal(shape=[t_dim]))


class FlowConditional(nn.Module):
    """Wraps an nflows Flow into the shared log_prob(theta, t) interface."""
    def __init__(self, kind, t_dim, n_params, hidden=64, n_layers=5,
                 num_bins=8, tail_bound=5.0, residual=True, num_blocks=2):
        super().__init__()
        self.kind = kind
        self.flow = _build_flow(kind, t_dim, n_params, hidden, n_layers,
                                num_bins, tail_bound, residual, num_blocks)

    def log_prob(self, theta, t):
        return self.flow.log_prob(inputs=t, context=theta)


def build_nle(nc, t_dim, n_params):
    kind = nc.get("model", "gmm").lower()
    if kind == "gmm":
        return kind, GMMConditional(
            t_dim, n_params,
            n_mix=int(nc.get("n_mix", 4)),
            hidden=int(nc.get("hidden", 64)),
            param_style=nc.get("param_style", "softplus"),
            diag_floor=float(nc.get("diag_floor", 1e-4)),
            legacy_branch=bool(nc.get("legacy_branch", False)),
            mean_style=nc.get("mean_style", "linear"))
    from sbi.nle import FlowConditional
    return kind, FlowConditional(kind, t_dim, n_params,
                                 hidden=int(nc.get("hidden", 64)),
                                 n_layers=int(nc.get("n_layers", 5)),
                                 num_bins=int(nc.get("num_bins", 8)),
                                 tail_bound=float(nc.get("tail_bound", 5.0)),
                                 residual=bool(nc.get("residual", True)),
                                 num_blocks=int(nc.get("num_blocks", 2)))


def nle_subdir(nle_root, nc):
    scope = "standard_t" if bool(nc.get("standardize", True)) else "raw_t"
    d = os.path.join(str(nle_root), scope,
                     nc.get("model", "gmm").lower() + str(nc.get("subdir_suffix", "")))
    if nc.get("seed_subdir", False):
        init_seed = int(nc.get("init_seed", nc.get("seed", 42)))
        d = os.path.join(d, f"seed_{init_seed}")
    os.makedirs(d, exist_ok=True)
    return d


def save_nle(model, kind, subdir, nc, t_dim, n_params, t_mean, t_std):
    torch.save(model.state_dict(), os.path.join(subdir, "nle_model.pt"))
    np.save(os.path.join(subdir, "t_mean.npy"), t_mean)
    np.save(os.path.join(subdir, "t_std.npy"), t_std)
    cfg = {"model": kind, "t_dim": t_dim, "n_params": n_params,
           "n_mix": int(nc.get("n_mix", 4)), "hidden": int(nc.get("hidden", 64)),
           "n_layers": int(nc.get("n_layers", 5)), "num_bins": int(nc.get("num_bins", 8)),
           "tail_bound": float(nc.get("tail_bound", 5.0)),
           "residual": bool(nc.get("residual", True)),
           "num_blocks": int(nc.get("num_blocks", 2)),
           "param_style": nc.get("param_style", "softplus"),
           "diag_floor": float(nc.get("diag_floor", 1e-4)),
           "legacy_branch": bool(nc.get("legacy_branch", False)),
           "mean_style": nc.get("mean_style", "linear")}
    with open(os.path.join(subdir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_nle(subdir, device="cpu"):
    with open(os.path.join(subdir, "model_config.json")) as f:
        cfg = json.load(f)
    if cfg.get("param_style") == "semelin_legacy" and "legacy_branch" not in cfg:
        cfg["param_style"] = "semelin"
    _, model = build_nle(cfg, cfg["t_dim"], cfg["n_params"])
    model.load_state_dict(torch.load(os.path.join(subdir, "nle_model.pt"),
                                     map_location=device))
    model.to(device).eval()
    t_mean = np.load(os.path.join(subdir, "t_mean.npy"))
    t_std = np.load(os.path.join(subdir, "t_std.npy"))
    return model, t_mean, t_std
