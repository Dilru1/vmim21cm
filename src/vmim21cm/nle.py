"""Stage-2 neural likelihood q(t | theta): masked autoregressive rational-quadratic spline flow."""
import json
import os

import numpy as np
import torch
import torch.nn as nn


class NSFLikelihood(nn.Module):
    """nflows NSF on standardised t, conditioned on theta."""

    def __init__(self, t_dim, n_params, hidden, n_layers, num_bins, tail_bound):
        super().__init__()
        from nflows import distributions, flows, transforms
        from nflows.transforms.autoregressive import \
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform as RQS
        tl = []
        for _ in range(n_layers):
            tl.append(RQS(features=t_dim, hidden_features=hidden, context_features=n_params,
                          num_blocks=2, num_bins=num_bins, tails="linear",
                          tail_bound=tail_bound, use_residual_blocks=True))
            tl.append(transforms.RandomPermutation(features=t_dim))
        self.flow = flows.Flow(transform=transforms.CompositeTransform(tl),
                               distribution=distributions.StandardNormal(shape=[t_dim]))

    def log_prob(self, theta, t):
        """log q(t | theta) per row."""
        return self.flow.log_prob(inputs=t, context=theta)


def build_nle(nc, t_dim, n_params):
    """NSF from the nle: block (hidden, n_layers, num_bins, tail_bound)."""
    return NSFLikelihood(t_dim, n_params, int(nc["hidden"]), int(nc["n_layers"]),
                         int(nc["num_bins"]), float(nc["tail_bound"]))


def save_nle(model, d, nc, t_dim, n_params, t_mean, t_std):
    """Write weights, t standardisation and model_config.json to d."""
    torch.save(model.state_dict(), os.path.join(d, "nle_model.pt"))
    np.save(os.path.join(d, "t_mean.npy"), t_mean)
    np.save(os.path.join(d, "t_std.npy"), t_std)
    meta = {k: nc[k] for k in ("hidden", "n_layers", "num_bins", "tail_bound")}
    meta.update(t_dim=int(t_dim), n_params=int(n_params))
    with open(os.path.join(d, "model_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_nle(d, device="cpu"):
    """-> (model, n_params, t_mean, t_std) from a stage-2 directory."""
    with open(os.path.join(d, "model_config.json")) as f:
        meta = json.load(f)
    model = build_nle(meta, meta["t_dim"], meta["n_params"])
    model.load_state_dict(torch.load(os.path.join(d, "nle_model.pt"), map_location=device))
    model.to(device).eval()
    return (model, meta["n_params"], np.load(os.path.join(d, "t_mean.npy")).astype(np.float32),
            np.load(os.path.join(d, "t_std.npy")).astype(np.float32))
