#!/usr/bin/env python3
"""Pluggable stage-1 compression objectives."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Objective(nn.Module):
    needs_head = False

    def forward(self, t, theta, aux=None, epoch=0):
        raise NotImplementedError

    def summary(self, t):
        return t

    def out_dim(self, t_dim):
        return t_dim

    @staticmethod
    def _warm(epoch, c):
        warm = int(c.get("warmup_epochs", 0))
        lam = float(c.get("lam_aux", 1.0))
        if warm <= 0:
            return 0.0
        if not bool(c.get("aux_anneal", False)):
            return lam if epoch < warm else 0.0
        ann = int(c.get("aux_anneal_epochs", warm))
        if epoch < warm:
            return lam
        if epoch < warm + ann:
            return lam * (1.0 - (epoch - warm) / max(1, ann))
        return 0.0


class MSEObjective(Objective):
    """Supervised regression -> f(x) approximates E[theta|x]: keeps the LOCATION of the likelihood ..."""

    def __init__(self, c, cfg, **kw):
        super().__init__()
        self.lam = float(c.get("lam_aux", 1.0))

    def forward(self, t, theta, aux=None, epoch=0):
        pred = aux if aux is not None else t
        loss = self.lam * F.mse_loss(pred, theta)
        return loss, {"mse": float(loss.detach())}


class VMIMObjective(Objective):
    """Barber-Agakov:  I(theta;t) >= E[log q_phi(theta|t)] + H(theta)."""
    needs_head = True

    def __init__(self, c, cfg, head=None, **kw):
        super().__init__()
        assert head is not None, "vmim needs a density head"
        self.head, self.c = head, c

    def forward(self, t, theta, aux=None, epoch=0):
        with torch.autocast("cuda", enabled=False):
            nll = -self.head.log_prob(t.float(), theta.float()).mean()
        logs = {"vmim": float(nll.detach())}
        w = self._warm(epoch, self.c)
        if w > 0 and aux is not None:
            m = F.mse_loss(aux, theta)
            logs["mse"] = float(m.detach())
            return nll + w * m, logs
        return nll, logs


class InfoNCEObjective(Objective):
    """Contrastive MI bound (van den Oord+18 / CPC) with a THETA NEGATIVE QUEUE."""

    def __init__(self, c, cfg, head=None, t_dim=None, n_params=None, **kw):
        super().__init__()
        t_dim = int(t_dim if t_dim is not None else cfg["t_dim"])
        n_params = int(n_params if n_params is not None else cfg["n_params"])
        d = int(c.get("nce_dim", 64))
        h = int(c.get("nce_hidden", 128))
        self.g = nn.Sequential(nn.Linear(t_dim, h), nn.SiLU(), nn.Linear(h, d))
        self.h = nn.Sequential(nn.Linear(n_params, h), nn.SiLU(), nn.Linear(h, d))
        self.log_tau = nn.Parameter(torch.tensor(float(c.get("nce_log_tau", -2.3))))
        self.symmetric = bool(c.get("nce_symmetric", True))
        self.c = c

        self.K = int(c.get("nce_queue", 8192))
        if self.K > 0:
            self.register_buffer("queue", torch.zeros(self.K, n_params))
            self.register_buffer("qptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("qfill", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _push(self, theta):
        n = theta.shape[0]
        if self.K <= 0 or n == 0:
            return
        p = int(self.qptr.item())
        idx = (torch.arange(n, device=theta.device) + p) % self.K
        self.queue[idx] = theta.detach().to(self.queue.dtype)
        self.qptr[0] = (p + n) % self.K
        self.qfill[0] = min(int(self.qfill.item()) + n, self.K)

    def forward(self, t, theta, aux=None, epoch=0):
        with torch.autocast("cuda", enabled=False):
            zt = F.normalize(self.g(t.float()), dim=-1)
            pos = theta.float()
            if self.K > 0 and int(self.qfill.item()) > 0 and self.training:
                neg = self.queue[: int(self.qfill.item())].to(pos.dtype)
                allth = torch.cat([pos, neg], 0)
            else:
                allth = pos
            zh = F.normalize(self.h(allth), dim=-1)
            S = zt @ zh.t() / self.log_tau.exp().clamp(min=1e-3)
            tgt = torch.arange(len(zt), device=S.device)
            loss = F.cross_entropy(S, tgt)
            if self.symmetric:
                loss = 0.5 * (loss + F.cross_entropy(S[:, : len(zt)].t(), tgt))
        if self.training:
            self._push(theta)

        n_eff = S.shape[1]
        logs = {"infonce": float(loss.detach()),
                "I_lb": float(math.log(n_eff) - loss.detach()),
                "log_N": math.log(n_eff),
                "n_eff": n_eff,
                "tau": float(self.log_tau.exp().detach())}
        w = self._warm(epoch, self.c)
        if w > 0 and aux is not None:
            m = F.mse_loss(aux, theta)
            logs["mse"] = float(m.detach())
            return loss + w * m, logs
        return loss, logs


class VIBObjective(Objective):
    """Variational information bottleneck:  max I(t;theta) - beta I(t;x)"""
    needs_head = True

    def __init__(self, c, cfg, head=None, **kw):
        super().__init__()
        assert head is not None, "vib needs a density head"
        self.head = head
        self.beta = float(c.get("vib_beta", 1e-3))
        self.c = c

    def summary(self, t):
        return t[:, : t.shape[1] // 2]

    def out_dim(self, t_dim):
        return t_dim // 2

    def forward(self, t, theta, aux=None, epoch=0):
        d = t.shape[1] // 2
        mu, logvar = t[:, :d], t[:, d:].clamp(-8.0, 8.0)
        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z = mu
        with torch.autocast("cuda", enabled=False):
            nll = -self.head.log_prob(z.float(), theta.float()).mean()
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl = kl_dim.sum(1).mean()
        loss = nll + self.beta * kl
        logs = {"vmim": float(nll.detach()), "kl_nats": float(kl.detach()),
                "beta": self.beta,
                "kl_per_dim": [round(float(v), 4) for v in kl_dim.mean(0).detach()]}
        w = self._warm(epoch, self.c)
        if w > 0 and aux is not None:
            m = F.mse_loss(aux, theta)
            logs["mse"] = float(m.detach())
            loss = loss + w * m
        return loss, logs


class IMNNObjective(Objective):
    """Information-maximising compression (Charnock, Lavaux & Wandelt 2018), with the two Fisher ..."""

    def __init__(self, c, cfg, head=None, t_dim=None, n_params=None, **kw):
        super().__init__()
        self.n_rep = int(c.get("imnn_n_rep", 4))
        self.lam = float(c.get("imnn_lambda", 10.0))
        self.ridge = float(c.get("imnn_ridge", 1e-4))
        self.c = c
        if self.n_rep < 2:
            raise ValueError("imnn_n_rep must be >= 2 (C is a within-sim covariance)")

    def forward(self, t, theta, aux=None, epoch=0):
        r = self.n_rep
        n, d = t.shape
        if n % r:
            raise RuntimeError(
                f"IMNN batch {n} is not a multiple of imnn_n_rep={r}. The loader "
                f"must be sbi.cubes_imnn.prepare_imnn_loaders with a matching "
                f"imnn_n_rep, and drop_last=True.")
        s = n // r

        with torch.autocast("cuda", enabled=False):
            T = t.float().view(s, r, d)
            mu = T.mean(1)
            dev = T - mu.unsqueeze(1)
            C = torch.einsum("srd,sre->de", dev, dev) / (s * (r - 1))
            C = C + self.ridge * torch.eye(d, device=C.device, dtype=C.dtype)

            TH = theta.float().view(s, r, -1)[:, 0, :]
            p = TH.shape[1]
            X = torch.cat([TH - TH.mean(0), torch.ones(s, 1, device=TH.device)], 1)
            J = torch.linalg.lstsq(X, mu - mu.mean(0)).solution[:p]

            Fish = J @ torch.linalg.solve(C, J.transpose(0, 1))
            Fish = Fish + self.ridge * torch.eye(p, device=Fish.device,
                                                 dtype=Fish.dtype)
            sign, logdet = torch.linalg.slogdet(Fish)
            logdet = torch.where(sign > 0, logdet,
                                 torch.full_like(logdet, -30.0))

            reg = ((C - torch.eye(d, device=C.device, dtype=C.dtype)) ** 2).sum()
            loss = -logdet + self.lam * reg

        logs = {"imnn_logdetF": float(logdet.detach()),
                "imnn_reg": float(reg.detach()),
                "n_sim": s, "n_rep": r,
                "F_diag": [round(float(v), 4) for v in torch.diagonal(Fish).detach()]}
        w = self._warm(epoch, self.c)
        if w > 0 and aux is not None:
            m = F.mse_loss(aux, theta)
            logs["mse"] = float(m.detach())
            loss = loss + w * m
        return loss, logs


class VMIMDirObjective(VMIMObjective):
    """VMIM + lam * 1D NLL along a fixed weak direction (added 2026-11-13)."""

    def __init__(self, c, cfg, head=None, **kw):
        super().__init__(c, cfg, head=head)
        from sbi.compressors.heads import VMIMGMMPosteriorHeadOLD
        if not isinstance(head, VMIMGMMPosteriorHeadOLD):
            raise ValueError("vmim_dir needs compressor.vmim_head: gmm_old")
        if getattr(head.pre, "jitter", None):
            raise ValueError("vmim_dir: dequantisation jitter not supported "
                             "(a second pre() call would draw different noise)")
        w = torch.tensor([float(x) for x in c["dir_w"]], dtype=torch.float32)
        sd = torch.tensor([float(x) for x in c["dir_prior_sd"]], dtype=torch.float32)
        if w.numel() != head.n_params or sd.numel() != head.n_params:
            raise ValueError("dir_w / dir_prior_sd must have length n_params")
        self.register_buffer("a", w / sd)
        self.lam = float(c.get("dir_lam", 1.0))
        print(f"[objective] vmim_dir lam={self.lam} w={w.tolist()} a={self.a.tolist()}")

    def dir_nll(self, t, theta):
        h = self.head
        th = h.pre(theta.float(), h.training)
        logits, means, tril = h.forward(t.float())
        L = h._L(tril, t.shape[0], t.device, torch.float32)
        a = self.a.to(t.device)
        mu_v = means @ a
        sd_v = ((L.transpose(-1, -2) @ a) ** 2).sum(-1).sqrt()
        lp = torch.distributions.Normal(mu_v, sd_v).log_prob((th @ a)[:, None])
        return -torch.logsumexp(torch.log_softmax(logits, dim=-1) + lp, dim=-1).mean()

    def forward(self, t, theta, aux=None, epoch=0):
        loss, logs = super().forward(t, theta, aux=aux, epoch=epoch)
        if self.lam == 0.0:
            return loss, logs
        with torch.autocast("cuda", enabled=False):
            nll_v = self.dir_nll(t, theta)
        logs["vdir"] = float(nll_v.detach())
        return loss + self.lam * nll_v, logs


_OBJECTIVES = {"mse": MSEObjective, "vmim": VMIMObjective,
               "vmim_dir": VMIMDirObjective,
               "infonce": InfoNCEObjective, "vib": VIBObjective,
               "imnn": IMNNObjective}


def objective_name(c):
    """Resolve the name WITHOUT building anything: stage 1 needs to know whether to construct a density ..."""
    if bool(c.get("aux_only", False)):
        return "mse"
    return str(c.get("objective", "vmim")).lower()


def needs_head(c):
    cls = _OBJECTIVES.get(objective_name(c))
    return bool(cls is not None and getattr(cls, "needs_head", False))


def build_objective(c, cfg, head=None, t_dim=None, n_params=None):
    name = objective_name(c)
    if name not in _OBJECTIVES:
        raise ValueError(f"unknown compressor.objective={name!r} "
                         f"(one of {sorted(_OBJECTIVES)})")
    return _OBJECTIVES[name](c, cfg, head=head, t_dim=t_dim, n_params=n_params)
