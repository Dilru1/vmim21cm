#!/usr/bin/env python3
"""Gaussian toy model for the SBC rank scores (Appendix C)."""
import os

import numpy as np
from scipy.stats import norm, skewnorm

import calib_ratio_toy_style as style

N, B, M, SEEDS = 908, 30, 2000, 50
EDGES = np.linspace(0, 1, B + 1)


def gauss(b, s):
    def draw(rng):
        z = rng.standard_normal(N)
        return z, b + s * rng.standard_normal((N, M))
    return draw, lambda r: norm.cdf(b + s * norm.ppf(r))


def theta_bias(tau):
    """bias varying across the test set, b_i ~ N(0, tau^2), true width."""
    def draw(rng):
        z = rng.standard_normal(N)
        bi = tau * rng.standard_normal(N)
        return z, bi[:, None] + rng.standard_normal((N, M))
    return draw, lambda r: norm.cdf(norm.ppf(r) / np.sqrt(1 + tau ** 2))


def skewed(alpha):
    """skew-normal posterior with the true mean and width: only the shape is wrong."""
    m, sd = skewnorm.mean(alpha), skewnorm.std(alpha)
    def draw(rng):
        z = rng.standard_normal(N)
        return z, (skewnorm.rvs(alpha, size=(N, M), random_state=rng) - m) / sd
    return draw, lambda r: norm.cdf((skewnorm.ppf(r, alpha) - m) / sd)


CASES = {
    "cal":   gauss(0.0, 1.0),
    "bias":  gauss(0.15, 1.0),
    "over":  gauss(0.0, 0.88),
    "under": gauss(0.0, 1.15),
    "both":  gauss(0.12, 0.92),
    "theta": theta_bias(0.3),
    "skew":  skewed(5.0),
}


def bin_probs(G):
    cdf = G(np.clip(EDGES, 1e-15, 1 - 1e-15))
    cdf[0], cdf[-1] = 0.0, 1.0
    return np.diff(cdf)


def expected_chi(p):
    """E[chi_hat] for N ranks with bin probabilities p (exact, multinomial)."""
    return np.sum(N * p * (1 - p) + (N * p - N / B) ** 2) / (N / B) / (B - 1)


def rank_scores(n):
    """Bias and width scores of a rank histogram n (B bin counts), each ~N(0, 1) when calibrated (linear ..."""
    n = np.asarray(n, float)
    B, N = len(n), n.sum()
    c = (np.arange(B) + 0.5) / B
    w = (c - 0.5) ** 2
    v0 = w.mean()
    zb = -(np.sum(n * c) / N - 0.5) / np.sqrt(v0 / N)
    zw = -(np.sum(n * w) / N - v0) / np.sqrt((np.mean(w ** 2) - v0 ** 2) / N)
    return zb, zw


def chi_contours():
    """Expected scores and E[chi_hat] on a (b, s) grid of the Gaussian model, to draw the chi_hat = 1.26 / ..."""
    pts = [(b, s) for b in np.linspace(-0.3, 0.3, 61) for s in np.linspace(0.75, 1.35, 61)]
    zz, chi = [], []
    for b, s in pts:
        p = bin_probs(gauss(b, s)[1])
        zz.append(rank_scores(N * p))
        chi.append(expected_chi(p))
    return np.array(zz), np.array(chi)


def main():
    scores, expected = {}, {}
    for i, (key, (draw, G)) in enumerate(CASES.items()):
        z = []
        for k in range(SEEDS):
            truth, post = draw(np.random.default_rng(10000 * (i + 1) + k))
            n, _ = np.histogram((post < truth[:, None]).mean(1), EDGES)
            z.append(rank_scores(n))
        scores[key] = np.array(z)
        p = bin_probs(G)
        expected[key] = np.array(rank_scores(N * p))
        mu, sd = scores[key].mean(0), scores[key].std(0, ddof=1)
        print(f"{key:>6s}  z_bias {mu[0]:+5.2f} +- {sd[0]:.2f} (expected {expected[key][0]:+5.2f})"
              f"   z_width {mu[1]:+5.2f} +- {sd[1]:.2f} (expected {expected[key][1]:+5.2f})"
              f"   E[chi_hat] {expected_chi(p):.2f}")
    fig_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "paper_plot_final", "images", "validation")
    os.makedirs(fig_out, exist_ok=True)
    out = os.path.join(fig_out, "calib_ratio_toy_scores.pdf")
    style.plot_scores(scores, expected, chi_contours(), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
