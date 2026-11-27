"""VMIM posterior heads, shared by both compressor arms."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms, distributions, flows


class ThetaPreprocessor(nn.Module):
    """Applied to theta inside log_prob, BEFORE any logit transform."""
    def __init__(self, jitter=None, margin=None, jitter_seed=0):
        super().__init__()
        self.jitter = dict(jitter or {})
        self.margin = dict(margin or {})
        self.jitter_seed = int(jitter_seed)
        self._gens = {}

    def set_jitter_seed(self, seed):
        self.jitter_seed = int(seed)
        self._gens = {}

    def _generator(self, device):
        key = str(device)
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.jitter_seed)
            self._gens[key] = g
        return g

    def forward(self, theta, training):
        if not self.jitter and not self.margin:
            return theta
        theta = theta.clone()
        if training and self.jitter:
            g = self._generator(theta.device)
            for d, half in self.jitter.items():
                col = theta[:, d]
                u = torch.rand(col.shape, generator=g,
                               device=col.device, dtype=col.dtype)
                theta[:, d] = col + (u * 2.0 - 1.0) * half
        for d, (a, b) in self.margin.items():
            theta[:, d] = a + b * theta[:, d]
        return theta


def _logit(theta, eps=1e-4):
    theta = theta.clamp(eps, 1.0 - eps)
    y = torch.log(theta) - torch.log1p(-theta)
    log_jac = (-torch.log(theta) - torch.log1p(-theta)).sum(-1)
    return y, log_jac


class FlowContextWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, context):
        return self.net(torch.cat([inputs, context], dim=-1))


class _Base(nn.Module):
    def __init__(self, t_dim, n_params, n_mix, hidden, sigma_floor=1e-2, pre=None,
                 jitter_seed=None):
        super().__init__()
        self.t_dim, self.n_params, self.n_mix = t_dim, n_params, n_mix
        if isinstance(sigma_floor, (list, tuple)):
            assert len(sigma_floor) == n_params, \
                f"sigma_floor list must have length n_params={n_params}, got {len(sigma_floor)}"
            floor_t = torch.tensor(sigma_floor, dtype=torch.float32)
        else:
            floor_t = torch.full((n_params,), float(sigma_floor), dtype=torch.float32)
        self.register_buffer("sigma_floor", floor_t)
        self.pre = pre if pre is not None else ThetaPreprocessor()
        if jitter_seed is not None:
            self.pre.set_jitter_seed(jitter_seed)
        self.net = nn.Sequential(nn.Linear(t_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, n_mix)
        self.means = nn.Linear(hidden, n_mix * n_params)
        self.raw_tril = nn.Linear(hidden, n_mix * (n_params * (n_params + 1) // 2))

    def forward(self, t):
        h = self.net(t)
        return self.logits(h), self.means(h).view(-1, self.n_mix, self.n_params), self.raw_tril(h)

    def _L(self, tril, B, device, dtype=None):
        dtype = tril.dtype if dtype is None else dtype
        L = torch.zeros(B, self.n_mix, self.n_params, self.n_params, device=device, dtype=dtype)
        idx = torch.tril_indices(self.n_params, self.n_params, 0, device=device)
        L[:, :, idx[0], idx[1]] = tril.view(B, self.n_mix, -1)
        di = torch.arange(self.n_params, device=device)
        floor = self.sigma_floor.to(device=device, dtype=dtype)
        L[:, :, di, di] = F.softplus(L[:, :, di, di]) + floor
        return L

    @torch.no_grad()
    def conditional_sigma(self, t):
        logits, means, tril = self.forward(t)
        w = torch.softmax(logits, dim=-1)
        L = self._L(tril, t.shape[0], t.device, t.dtype)
        comp_var = (L ** 2).sum(-1)
        mu = (w.unsqueeze(-1) * means).sum(1)
        second = (w.unsqueeze(-1) * (comp_var + means ** 2)).sum(1)
        return (second - mu ** 2).clamp_min(1e-12).sqrt()


class VMIMGMMPosteriorHeadOLD(_Base):
    """GMM over (pre-processed) theta, no logit."""
    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        logits, means, tril = self.forward(t)
        L = self._L(tril, t.shape[0], t.device, t.dtype)
        lp = torch.distributions.MultivariateNormal(loc=means, scale_tril=L).log_prob(theta[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + lp, dim=-1)


class VMIMGMMPosteriorHeadNEW(_Base):
    """GMM over logit(pre-processed theta)."""
    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        y, log_jac = _logit(theta)
        logits, means, tril = self.forward(t)
        L = self._L(tril, t.shape[0], t.device, t.dtype)
        lp = torch.distributions.MultivariateNormal(loc=means, scale_tril=L).log_prob(y[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + lp, dim=-1) + log_jac


class VMIMNFPosteriorHead(nn.Module):
    def __init__(self, t_dim, n_params, hidden=128, num_layers=4, num_bins=8,
                 tail_bound=4.0, logit=True, pre=None):
        super().__init__()
        from nflows.utils import create_alternating_binary_mask
        self.t_dim, self.n_params, self.logit = t_dim, n_params, logit
        self.pre = pre if pre is not None else ThetaPreprocessor()

        base_dist = distributions.StandardNormal(shape=[n_params])
        transform_list = []
        for i in range(num_layers):
            mask = create_alternating_binary_mask(features=n_params, even=(i % 2 == 0))

            def make_transform_net(in_features, out_features):
                net = nn.Sequential(
                    nn.Linear(in_features + t_dim, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, out_features),
                )
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                return FlowContextWrapper(net)

            transform_list.append(
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask, transform_net_create_fn=make_transform_net,
                    num_bins=num_bins, tails="linear", tail_bound=tail_bound,
                    apply_unconditional_transform=False))
            transform_list.append(transforms.RandomPermutation(features=n_params))

        self.flow = flows.Flow(transform=transforms.CompositeTransform(transform_list),
                               distribution=base_dist)

    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        if self.logit:
            y, log_jac = _logit(theta)
            return self.flow.log_prob(inputs=y, context=t) + log_jac
        return self.flow.log_prob(inputs=theta, context=t)


class VMIMMAFPosteriorHead(nn.Module):
    def __init__(self, t_dim, n_params, hidden=128, num_layers=4, num_blocks=2,
                 tail_bound=None, logit=True, pre=None):
        super().__init__()
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        self.t_dim, self.n_params, self.logit = t_dim, n_params, logit
        self.pre = pre if pre is not None else ThetaPreprocessor()

        base_dist = distributions.StandardNormal(shape=[n_params])
        transform_list = []
        for _ in range(num_layers):
            transform_list.append(
                MaskedAffineAutoregressiveTransform(
                    features=n_params, hidden_features=hidden, context_features=t_dim,
                    num_blocks=num_blocks, use_residual_blocks=True,
                    random_mask=False, activation=F.relu))
            transform_list.append(transforms.RandomPermutation(features=n_params))

        self.flow = flows.Flow(transform=transforms.CompositeTransform(transform_list),
                               distribution=base_dist)

    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        if self.logit:
            y, log_jac = _logit(theta)
            return self.flow.log_prob(inputs=y, context=t) + log_jac
        return self.flow.log_prob(inputs=theta, context=t)


def build_head(kind, t_dim, n_params, n_mix, hidden,
               sigma_floor=1e-2, jitter=None, margin=None, jitter_seed=0):
    """jitter/margin are {param_index: value} dicts, e.g. jitter={2: 0.05}, margin={2: (0.07, 0.86)} for ..."""
    pre = ThetaPreprocessor(jitter=jitter, margin=margin, jitter_seed=jitter_seed)
    if kind == "gmm_old":
        return VMIMGMMPosteriorHeadOLD(t_dim, n_params, n_mix, hidden,
                                       sigma_floor=sigma_floor, pre=pre,
                                       jitter_seed=jitter_seed)
    if kind == "gmm_new":
        return VMIMGMMPosteriorHeadNEW(t_dim, n_params, n_mix, hidden,
                                       sigma_floor=sigma_floor, pre=pre,
                                       jitter_seed=jitter_seed)
    if kind == "nf_spline":
        return VMIMNFPosteriorHead(t_dim=t_dim, n_params=n_params, hidden=hidden,
                                   num_layers=max(2, n_mix), num_bins=8, pre=pre)
    if kind == "nf_maf":
        return VMIMMAFPosteriorHead(t_dim=t_dim, n_params=n_params, hidden=hidden,
                                    num_layers=max(2, n_mix), num_blocks=2, pre=pre)
    raise ValueError(kind)


def build_head_from_cfg(c, cfg):
    """Convenience: read the new knobs straight from the compressor block."""
    jitter, margin = {}, {}

    if c.get("theta2_dequant", False):
        jitter[2] = float(c.get("theta2_grid_half", 0.05))
    if c.get("theta2_margin", False):
        h = jitter.get(2, 0.0)
        b = 0.96 / (1.0 + 2.0 * h)
        a = 0.02 + h * b
        margin[2] = (a, b)

    for idx, spec in c.get("dequant", {}).items():
        idx = int(idx)
        h = float(spec.get("half", 0.0))
        if h > 0:
            jitter[idx] = h
        if spec.get("margin", False):
            b = 0.96 / (1.0 + 2.0 * h)
            a = 0.02 + h * b
            margin[idx] = (a, b)

    raw_floor = c.get("sigma_floor", 1e-2)
    sigma_floor = list(raw_floor) if isinstance(raw_floor, (list, tuple)) else float(raw_floor)

    jitter_seed = int(c.get("jitter_seed", c.get("init_seed", c.get("seed", 0))))

    return build_head(c["vmim_head"], cfg["t_dim"], cfg["n_params"],
                      c["n_mix_head"], c["hidden_head"],
                      sigma_floor=sigma_floor,
                      jitter=jitter or None, margin=margin or None,
                      jitter_seed=jitter_seed)
