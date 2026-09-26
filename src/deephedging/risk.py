"""Convex risk measures used as training objectives.

Sign convention: the argument X is a loss, so training minimises rho(-PnL).
Larger is worse. Both measures are monetary, rho(X + a) = rho(X) + a, which is
what lets `pricing.py` read an indifference price off a difference of two risk
numbers.
"""

from __future__ import annotations

import torch


def entropic_risk(loss: torch.Tensor, lam: float) -> torch.Tensor:
    """rho(X) = (1/lambda) log E[exp(lambda X)].

    Computed with logsumexp: exp(lambda X) overflows float32 past lambda X ~ 88,
    which the untrained network hits on its first batch.
    """
    if lam <= 0.0:
        raise ValueError("lambda must be strictly positive; use the mean for the lambda -> 0 limit")
    n = loss.shape[0]
    log_mean_exp = torch.logsumexp(lam * loss, dim=0) - torch.log(
        torch.tensor(float(n), dtype=loss.dtype, device=loss.device)
    )
    return log_mean_exp / lam


def cvar(loss: torch.Tensor, alpha: float, w: torch.Tensor | None = None) -> torch.Tensor:
    """CVaR at level alpha in Rockafellar-Uryasev form,

        CVaR_alpha(X) = min_w { w + (1/(1 - alpha)) E[(X - w)^+] },

    whose minimiser is the VaR. Used instead of the plain quantile because the
    quantile of a sample is piecewise constant and has no useful gradient. In
    training `w` is a parameter optimised with the network; at evaluation time
    `w=None` plugs in the empirical VaR, which attains the minimum.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if w is None:
        w_hat = torch.quantile(loss, alpha)
        return w_hat + torch.clamp(loss - w_hat, min=0.0).mean() / (1.0 - alpha)
    return w + torch.clamp(loss - w, min=0.0).mean() / (1.0 - alpha)


def make_objective(kind: str, lam: float = 1.0, alpha: float = 0.95):
    """Return (objective, extra_parameters) for the training loop.

    `extra_parameters` is what the optimiser must see besides the network
    weights: empty for the entropic measure, the CVaR variable w for CVaR.
    """
    if kind == "entropic":

        def objective(loss: torch.Tensor) -> torch.Tensor:
            return entropic_risk(loss, lam)

        return objective, []

    if kind == "cvar":
        # The optimal w is the VaR, which sits within a few price units of zero.
        w = torch.zeros(1, requires_grad=True)

        def objective(loss: torch.Tensor) -> torch.Tensor:
            return cvar(loss, alpha, w=w[0])

        return objective, [w]

    raise ValueError(f"unknown risk measure {kind!r}; expected 'entropic' or 'cvar'")


def evaluate_risk(loss: torch.Tensor, kind: str, lam: float = 1.0, alpha: float = 0.95) -> float:
    """Risk of a loss sample, for reporting."""
    with torch.no_grad():
        if kind == "entropic":
            return float(entropic_risk(loss, lam))
        if kind == "cvar":
            return float(cvar(loss, alpha, w=None))
    raise ValueError(f"unknown risk measure {kind!r}")
