"""Indifference pricing: turning a risk number into a price.

    p = inf_delta rho(-PnL_with_claim) - inf_delta rho(-PnL_without_claim).

The second term is zero here: with no claim there is nothing to hedge, the paths
carry no expected return and costs strictly lose, so staying flat is optimal and
rho(0) = 0. The price is therefore the optimal value of the training objective.

Two tested consequences. At zero cost, lambda -> 0 recovers the Black-Scholes
price, since the entropic measure tends to E[-PnL] and the hedging gain has mean
zero for any predictable strategy. And since
rho_lambda(X) = E[X] + (lambda / 2) Var(X) + O(lambda^2), a linear fit in lambda
gives the price as the intercept and the unhedgeable residual variance as the
slope.
"""

from __future__ import annotations

import numpy as np
import torch

from deephedging.hedging import pnl
from deephedging.risk import evaluate_risk
from deephedging.train import _to_tensor


def indifference_price(
    S: np.ndarray,
    deltas: np.ndarray,
    strike: float,
    cost_rate: float = 0.0,
    risk: str = "entropic",
    lam: float = 1.0,
    alpha: float = 0.95,
) -> float:
    """rho(-PnL) for a given strategy, which is the indifference price when the
    strategy is optimal and the no-claim optimum is to stay flat."""
    with torch.no_grad():
        loss = -pnl(_to_tensor(S), _to_tensor(deltas), strike, cost_rate)
    return evaluate_risk(loss.double(), risk, lam=lam, alpha=alpha)


def fit_lambda_intercept(lambdas: np.ndarray, prices: np.ndarray) -> tuple[float, float]:
    """Least-squares line through (lambda, price); returns (intercept, slope).

    The intercept estimates the lambda -> 0 limit using every point, rather than
    the smallest lambda alone where the Monte Carlo error is worst.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    prices = np.asarray(prices, dtype=float)
    A = np.stack([np.ones_like(lambdas), lambdas], axis=1)
    coef, *_ = np.linalg.lstsq(A, prices, rcond=None)
    return float(coef[0]), float(coef[1])
