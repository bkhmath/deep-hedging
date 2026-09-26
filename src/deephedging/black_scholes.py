"""Closed-form Black-Scholes pricing and the implied-volatility inversion.

This is the case where the answer is already known: frictionless and
continuously rebalanced, the delta hedge replicates the call exactly, so a
network that does not converge to it means the training code is wrong.
Vectorised over numpy arrays, since the calibration prices a whole surface.

    S, K     spot and strike        r, q    rate and dividend yield
    T        maturity in years      sigma   annualised volatility
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The two standardised log-moneyness arguments of the Black-Scholes formula."""
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def call_price(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float = 0.0,
    q: float = 0.0,
    sigma: np.ndarray | float = 0.2,
) -> np.ndarray:
    """European call price.

    The T == 0 and sigma == 0 edges are handled separately rather than left to
    divide by zero: the tests probe the payoff limit, and the calibrator can
    walk to sigma near zero.
    """
    S, K, T, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    intrinsic = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    degenerate = (T <= 0.0) | (sigma <= 0.0)

    # np.where evaluates both branches, so feed the formula a harmless value on
    # the degenerate entries instead of letting it emit warnings.
    T_safe = np.where(degenerate, 1.0, T)
    sigma_safe = np.where(degenerate, 1.0, sigma)
    d1, d2 = _d1_d2(S, K, T_safe, r, q, sigma_safe)
    smooth = S * np.exp(-q * T_safe) * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(degenerate, intrinsic, smooth)


def call_delta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float = 0.0,
    q: float = 0.0,
    sigma: np.ndarray | float = 0.2,
) -> np.ndarray:
    """dC/dS for a European call, the position the replicating portfolio holds.

    The benchmark hedge, derived assuming trading is free. Once it costs
    something this stops being optimal, which is what the study measures.
    """
    S, K, T, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    # At expiry the delta is the indicator of being in the money.
    limit = np.exp(-q * np.maximum(T, 0.0)) * (S > K).astype(float)

    T_safe = np.where(degenerate, 1.0, T)
    sigma_safe = np.where(degenerate, 1.0, sigma)
    d1, _ = _d1_d2(S, K, T_safe, r, q, sigma_safe)
    return np.where(degenerate, limit, np.exp(-q * T_safe) * norm.cdf(d1))


def call_vega(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float = 0.0,
    q: float = 0.0,
    sigma: np.ndarray | float = 0.2,
) -> np.ndarray:
    """dC/dsigma. Used to weight the calibration residuals, not to hedge."""
    S, K, T, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    T_safe = np.where(degenerate, 1.0, T)
    sigma_safe = np.where(degenerate, 1.0, sigma)
    d1, _ = _d1_d2(S, K, T_safe, r, q, sigma_safe)
    vega = S * np.exp(-q * T_safe) * norm.pdf(d1) * np.sqrt(T_safe)
    return np.where(degenerate, 0.0, vega)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    q: float = 0.0,
    lo: float = 1e-4,
    hi: float = 10.0,
) -> float:
    """Invert `call_price` in sigma for a single quote.

    Brent on [lo, hi] rather than Newton: the price is strictly increasing in
    sigma, so bracketing cannot fail, while Newton can step out of the domain
    near the no-arbitrage bounds. Returns NaN for a price outside the
    arbitrage-free range, so a bad quote drops out instead of poisoning the fit.
    """
    lower_bound = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper_bound = S * np.exp(-q * T)
    if not (lower_bound < price < upper_bound):
        return float("nan")

    def excess(sigma: float) -> float:
        return float(call_price(S, K, T, r, q, sigma)) - price

    if excess(lo) > 0.0 or excess(hi) < 0.0:
        return float("nan")
    return float(brentq(excess, lo, hi, xtol=1e-10, maxiter=200))
