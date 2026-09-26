"""The analytic hedging strategies the learned one is measured against.

The Black-Scholes delta, the Heston model delta, and the Heston
minimum-variance delta. None of them knows about transaction costs.

Under stochastic volatility the model delta leaves the variance exposure
unhedged, and part of it is hedgeable with the stock because the two Brownians
are correlated. Minimising the instantaneous variance over the share count,

    delta_mv = dC/dS + (rho * xi / S) * dC/dv,

which holds fewer shares than the model delta when rho < 0. Without it the
network would get credit for recovering this classical correction.

The Greeks are tabulated and interpolated rather than quadratured per path and
date, on a grid in standardised log-moneyness x = log(S/K) / sqrt(w), with
w = theta*tau + (v - theta)(1 - e^{-kappa tau})/kappa. There the delta is
approximately Phi(x), smooth at every maturity; a uniform grid in S/K cannot
resolve it near expiry and errs by 0.23.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from deephedging import black_scholes as bs
from deephedging import heston
from deephedging.heston import HestonParams


def bs_delta_positions(
    S: np.ndarray,
    strike: float,
    maturity: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Black-Scholes delta at every date on every path, (n_paths, n_steps).

    Only the first n_steps columns of S are used: at maturity the hedge is
    liquidated, not reset.
    """
    n_steps = S.shape[1] - 1
    dt = maturity / n_steps
    tau = maturity - dt * np.arange(n_steps)  # time to maturity at each decision date
    return bs.call_delta(S[:, :n_steps], strike, tau[None, :], r, q, sigma)


def integrated_variance(tau: np.ndarray, v: np.ndarray, p: HestonParams) -> np.ndarray:
    """w = integral over [t, T] of E[v_s | v_t = v] ds, in closed form.

    Since E[v_s] = theta + (v - theta) e^{-kappa (s - t)}, this is
    theta*tau + (v - theta)(1 - e^{-kappa tau})/kappa: the Black-Scholes
    equivalent total variance.
    """
    decay = np.where(
        p.kappa * tau > 1e-8,
        (1.0 - np.exp(-p.kappa * tau)) / p.kappa,
        tau,  # the kappa -> 0 limit, to avoid 0/0 at the last date
    )
    return np.maximum(p.theta * tau + (v - p.theta) * decay, 1e-12)


@lru_cache(maxsize=8)
def _heston_grids(
    maturity: float,
    n_steps: int,
    p: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    n_x: int = 97,
    n_variance: int = 25,
    x_max: float = 8.0,
    variance_max_factor: float = 8.0,
):
    """Tabulate the Heston delta and dC/dv on a (tau, x, variance) grid.

    Cached because the study asks for the model delta and the minimum-variance
    delta of the same model; `HestonParams` is frozen and so hashable.

    dC/dv is a central difference in the initial variance at a fixed strike.
    Differentiating the interpolated price in v would also pick up the
    dependence of the standardised coordinate on v.
    """
    dt = maturity / n_steps
    taus = maturity - dt * np.arange(n_steps)
    xs = np.linspace(-x_max, x_max, n_x)
    v_max = variance_max_factor * max(p.v0, p.theta)
    variances = np.linspace(1e-6, v_max, n_variance)

    delta = np.empty((n_steps, n_x, n_variance))
    dprice_dv = np.empty((n_steps, n_x, n_variance))
    h = 0.25 * (variances[1] - variances[0])

    for i, tau in enumerate(taus):
        for j, v in enumerate(variances):
            # Spot is fixed at one and the strikes are chosen so that the
            # standardised log-moneyness runs over the grid. Homogeneity of degree
            # one in (S, K) makes this table valid at any spot.
            w = float(integrated_variance(np.array(tau), np.array(v), p))
            strikes = np.exp(-xs * np.sqrt(w))

            pv = HestonParams(float(v), p.kappa, p.theta, p.xi, p.rho)
            delta[i, :, j] = heston.call_delta(1.0, strikes, float(tau), r, q, pv)

            up = HestonParams(float(v) + h, p.kappa, p.theta, p.xi, p.rho)
            down = HestonParams(max(float(v) - h, 1e-9), p.kappa, p.theta, p.xi, p.rho)
            c_up = heston.call_price(1.0, strikes, float(tau), r, q, up)
            c_down = heston.call_price(1.0, strikes, float(tau), r, q, down)
            dprice_dv[i, :, j] = (c_up - c_down) / (up.v0 - down.v0)

    # taus is decreasing and RegularGridInterpolator needs ascending axes.
    order = np.argsort(taus)
    grid = (taus[order], xs, variances)
    delta_interp = RegularGridInterpolator(grid, delta[order], bounds_error=False, fill_value=None)
    dv_interp = RegularGridInterpolator(grid, dprice_dv[order], bounds_error=False, fill_value=None)
    return delta_interp, dv_interp, taus, x_max


def heston_delta_positions(
    S: np.ndarray,
    v: np.ndarray,
    strike: float,
    maturity: float,
    p: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    minimum_variance: bool = False,
) -> np.ndarray:
    """Heston delta, or minimum-variance delta, on every path and date."""
    n_steps = S.shape[1] - 1
    delta_interp, dv_interp, taus, x_max = _heston_grids(maturity, n_steps, p, r, q)

    tau_col = np.repeat(taus[None, :], S.shape[0], axis=0)
    variance = v[:, :n_steps]
    w = integrated_variance(tau_col, variance, p)
    # Clipped to the tabulated range: beyond eight standard deviations the delta is
    # zero or one to machine precision, so the clip loses nothing and it stops the
    # interpolator from extrapolating where the variance has collapsed to zero.
    x = np.clip(np.log(S[:, :n_steps] / strike) / np.sqrt(w), -x_max, x_max)

    points = np.stack([tau_col.ravel(), x.ravel(), variance.ravel()], axis=1)
    delta = delta_interp(points).reshape(x.shape)
    if minimum_variance:
        # The table holds dC/dv for a spot of one; scale by S, then the formula's
        # own 1/S cancels it. Written out instead of cancelled, so the expression
        # matches the derivation in the module docstring.
        dC_dv = dv_interp(points).reshape(x.shape) * S[:, :n_steps]
        delta = delta + (p.rho * p.xi / S[:, :n_steps]) * dC_dv
    return delta
