"""Path simulation for the two market models.

Simulated under the pricing measure (drift r - q, no risk premium), so the
hedging gain has expectation zero and the indifference price converges to
Black-Scholes as risk aversion goes to zero. The hedging experiments take
r = q = 0, which removes the cash-accrual term from the wealth equation.

Paths are (n_paths, n_steps + 1) numpy arrays including both endpoints, so
column k is the state at date t_k.
"""

from __future__ import annotations

import numpy as np

from deephedging.heston import HestonParams


def gbm_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    S0: float = 100.0,
    sigma: float = 0.2,
    r: float = 0.0,
    q: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Geometric Brownian motion on a uniform grid.

    Exact: log increments of a GBM are Gaussian, so there is no scheme bias and
    the only discretisation is the hedging grid itself.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
        (n_paths, n_steps)
    )
    log_paths = np.concatenate(
        [np.full((n_paths, 1), np.log(S0)), np.cumsum(increments, axis=1)], axis=1
    )
    log_paths[:, 1:] += np.log(S0)
    return np.exp(log_paths)


def heston_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    p: HestonParams,
    S0: float = 100.0,
    r: float = 0.0,
    q: float = 0.0,
    seed: int = 0,
    substeps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Heston paths by full-truncation Euler (Lord, Koekkoek, van Dijk, 2010).

    Returns (S, v), each (n_paths, n_steps + 1), on the rebalancing grid.
    `substeps` refines each rebalancing interval internally and reports only the
    endpoints, so changing the hedging frequency does not change the simulated
    market: Euler accuracy is governed by kappa * dt, and the calibrated BTC
    surface has kappa near 20.

    Full truncation keeps the possibly-negative v as the state but evaluates the
    coefficients at v^+ = max(v, 0). Needed because the naive Euler step gives
    v < 0 with positive probability whenever the Feller condition
    2 kappa theta >= xi^2 fails, which it does on the calibrated parameters, and
    the next sqrt(v) is then NaN. Lord et al. show this variant has the smallest
    bias of the standard repairs. The log price is stepped so S stays positive.
    """
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    rng = np.random.default_rng(seed)
    n_fine = n_steps * substeps
    dt = T / n_fine
    sqrt_dt = np.sqrt(dt)

    log_s_obs = np.empty((n_paths, n_steps + 1))
    v_obs = np.empty((n_paths, n_steps + 1))
    log_s_obs[:, 0] = np.log(S0)
    v_obs[:, 0] = p.v0

    log_s = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, p.v0)  # the raw, possibly negative, scheme state

    for i in range(n_fine):
        # Correlated normals by Cholesky of [[1, rho], [rho, 1]], drawn one fine
        # step at a time so memory scales with paths and not with the fine grid.
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_v = p.rho * z1 + np.sqrt(1.0 - p.rho**2) * z2

        v_plus = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_plus)
        log_s = log_s + (r - q - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z1
        v = v + p.kappa * (p.theta - v_plus) * dt + p.xi * sqrt_v * sqrt_dt * w_v

        if (i + 1) % substeps == 0:
            k = (i + 1) // substeps
            log_s_obs[:, k] = log_s
            v_obs[:, k] = v

    # The network must receive a variance, so report v^+; the untruncated v is
    # internal to the scheme.
    return np.exp(log_s_obs), np.maximum(v_obs, 0.0)


def call_payoff(S_T: np.ndarray, K: float) -> np.ndarray:
    """(S_T - K)^+, the claim being hedged."""
    return np.maximum(S_T - K, 0.0)
