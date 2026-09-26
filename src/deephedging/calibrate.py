"""Calibrating Heston to an option surface.

Five parameters against a few hundred quotes, non-convex, and not equally
identified: the surface pins v0 and rho tightly, while kappa and theta enter
mostly through kappa*theta and through the thin long end. Eight starting points
are tried and the spread of the local optima reported; on both surfaces used
here all eight reach the same optimum to twelve digits.

The real pathology on the BTC surface is in the parameters, not the spread: the
fit drives kappa to its bound of 20 and xi to nearly 4, badly violating Feller,
because the short-dated BTC smile is more convex than Heston can otherwise
produce. A parameter on a bound means the model is misspecified for the data.

The objective is vega-weighted price residuals. Since

    iv_model - iv_market = (C_model - C_market) / vega + O(.^2),

dividing by the market vega gives the implied-vol residual to first order with
no Black-Scholes inversion in the loop, and with vega fixed per quote the
objective cannot game it. Fit quality is inverted once at the end and reported
in genuine implied-vol space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from deephedging import black_scholes as bs
from deephedging import heston
from deephedging.heston import HestonParams

# (v0, kappa, theta, xi, rho)
LOWER = np.array([1e-4, 0.1, 1e-4, 1e-2, -0.995])
UPPER = np.array([4.0, 20.0, 4.0, 10.0, 0.995])


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_iv: float  # root mean squared implied-vol error, in vol points
    max_abs_iv: float
    n_quotes: int
    feller: float
    starts_tried: int
    objective_spread: float  # spread of the local optima found, a look at identifiability
    n_dropped: int  # quotes excluded for carrying no volatility information
    fit: pd.DataFrame  # per-quote market and model implied vols

    def summary(self) -> str:
        p = self.params
        if abs(self.feller) < 1e-3:
            # A constrained fit pushes it to exactly zero; calling that
            # "violated" on the sign of a rounding error would mislead.
            feller_note = "binding"
        else:
            feller_note = "satisfied" if self.feller > 0 else "VIOLATED"
        parts = [
            f"v0={p.v0:.4f}  kappa={p.kappa:.3f}  theta={p.theta:.4f}  "
            f"xi={p.xi:.3f}  rho={p.rho:+.3f}",
            f"implied-vol RMSE {self.rmse_iv * 100:.2f} vol points, "
            f"max |error| {self.max_abs_iv * 100:.2f} vol points, over {self.n_quotes} quotes",
            f"Feller 2*kappa*theta - xi^2 = {self.feller:+.4f} ({feller_note})",
        ]
        if self.n_dropped:
            parts.append(f"{self.n_dropped} quotes dropped for negligible vega")
        return "\n".join(parts)


def _model_prices(
    params: np.ndarray,
    forwards: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Heston prices for the surface, one quadrature pass per maturity slice.

    Each slice is priced in its own forward measure: spot is that slice's
    forward, rate and carry zero. See `deribit.py` for why that is exact.
    """
    p = HestonParams(*params)
    out = np.empty(len(strikes))
    for T in np.unique(maturities):
        mask = maturities == T
        F = forwards[mask][0]
        out[mask] = heston.call_price(float(F), strikes[mask], float(T), 0.0, 0.0, p, n_nodes)
    return out


def _residuals(
    params: np.ndarray,
    forwards: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    vegas: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    model = _model_prices(params, forwards, strikes, maturities, n_nodes)
    return weights * (model - market_prices) / vegas


def _starting_points() -> np.ndarray:
    """Eight starting points, fixed rather than random so the fit is
    reproducible. They differ in initial variance, mean-reversion speed and
    skew."""
    return np.array(
        [
            [0.20, 2.0, 0.20, 1.0, -0.50],
            [0.50, 1.0, 0.40, 1.5, -0.30],
            [0.10, 5.0, 0.15, 0.8, -0.70],
            [0.80, 0.5, 0.80, 2.0, -0.10],
            [0.30, 8.0, 0.30, 2.5, -0.60],
            [0.15, 1.5, 0.60, 0.5, +0.20],
            [0.40, 3.0, 0.10, 1.2, -0.85],
            [0.25, 0.3, 0.25, 0.6, -0.40],
        ]
    )


def calibrate(
    surface: pd.DataFrame,
    n_nodes: int = 128,
    enforce_feller: bool = False,
    n_nodes_report: int = 256,
    min_vega_fraction: float = 1e-4,
) -> CalibrationResult:
    """Fit Heston to a cleaned surface.

    `surface` must carry the columns produced by `deribit.clean_surface`:
    forward, strike, maturity, market_iv, rel_spread.

    `enforce_feller` penalises 2*kappa*theta - xi^2 < 0. Off by default: real
    surfaces violate it routinely, and full-truncation Euler copes. Set it to
    True to see what imposing the condition costs the fit.

    `min_vega_fraction` drops quotes whose vega is below that fraction of the
    forward. The objective divides the price residual by vega, so a near-zero
    vega turns a 1e-10 pricing error into a residual of order one; four such
    quotes were enough to pull the parameters off an optimum otherwise reached
    to four decimals. They carry no volatility information anyway. The liquidity
    filters in `deribit.clean_surface` already remove them from the real
    surface, so the threshold only bites on a synthetic grid.
    """
    forwards = surface["forward"].to_numpy(dtype=float)
    strikes = surface["strike"].to_numpy(dtype=float)
    maturities = surface["maturity"].to_numpy(dtype=float)
    market_iv = surface["market_iv"].to_numpy(dtype=float)

    # From the quoted implied vols rather than the quoted premium: keeps
    # everything in USD and sidesteps the BTC-settlement conversion.
    market_prices = bs.call_price(forwards, strikes, maturities, 0.0, 0.0, market_iv)
    vegas = bs.call_vega(forwards, strikes, maturities, 0.0, 0.0, market_iv)

    informative = (vegas > min_vega_fraction * forwards) & np.isfinite(market_iv)
    n_dropped = int((~informative).sum())
    if n_dropped:
        surface = surface.loc[informative].reset_index(drop=True)
        forwards, strikes = forwards[informative], strikes[informative]
        maturities, market_iv = maturities[informative], market_iv[informative]
        market_prices, vegas = market_prices[informative], vegas[informative]
    if len(strikes) < 10:
        raise ValueError(
            f"only {len(strikes)} quotes carry usable volatility information; that is "
            "too thin a surface to calibrate five parameters to"
        )

    # Clipped so that one implausibly tight quote cannot dominate the fit, and so
    # that a missing spread column falls back to equal weights.
    if "rel_spread" in surface:
        spread = surface["rel_spread"].to_numpy(dtype=float)
        weights = 1.0 / np.clip(spread, 0.05, 2.0)
    else:
        weights = np.ones_like(strikes)
    weights = weights / np.sqrt(np.mean(weights**2))  # scale-free, for readable costs

    def residual_fn(params: np.ndarray) -> np.ndarray:
        r = _residuals(
            params, forwards, strikes, maturities, market_prices, vegas, weights, n_nodes
        )
        if enforce_feller:
            p = HestonParams(*params)
            # One extra residual, active only when the condition is violated.
            r = np.append(r, 10.0 * min(p.feller(), 0.0))
        return r

    best = None
    costs = []
    for start in _starting_points():
        start = np.clip(start, LOWER, UPPER)
        try:
            fit = least_squares(
                residual_fn,
                start,
                bounds=(LOWER, UPPER),
                method="trf",
                xtol=1e-10,
                ftol=1e-10,
                max_nfev=600,
            )
        except Exception:
            continue
        costs.append(fit.cost)
        if best is None or fit.cost < best.cost:
            best = fit

    if best is None:
        raise RuntimeError("every starting point failed")

    params = HestonParams(*best.x)
    # Inverted once, at the end, at each slice's own forward. This is the
    # quantity reported; the vega-weighted objective is its first-order proxy.
    model_iv = np.empty(len(strikes))
    for T in np.unique(maturities):
        mask = maturities == T
        F = float(forwards[mask][0])
        prices = np.atleast_1d(
            heston.call_price(F, strikes[mask], float(T), 0.0, 0.0, params, n_nodes_report)
        )
        for j, i in enumerate(np.flatnonzero(mask)):
            model_iv[i] = bs.implied_vol(float(prices[j]), F, float(strikes[i]), float(T))

    fit_table = surface.copy()
    fit_table["model_iv"] = model_iv
    fit_table["iv_error"] = model_iv - market_iv
    valid = np.isfinite(model_iv)

    return CalibrationResult(
        params=params,
        rmse_iv=float(np.sqrt(np.mean(fit_table.loc[valid, "iv_error"] ** 2))),
        max_abs_iv=float(np.max(np.abs(fit_table.loc[valid, "iv_error"]))),
        n_quotes=int(valid.sum()),
        feller=params.feller(),
        starts_tried=len(costs),
        objective_spread=float(np.max(costs) - np.min(costs)) if costs else float("nan"),
        n_dropped=n_dropped,
        fit=fit_table,
    )


def synthetic_surface(
    params: HestonParams,
    forward: float = 100.0,
    maturities: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    moneyness: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5),
    n_nodes: int = 256,
) -> pd.DataFrame:
    """A surface generated from known Heston parameters.

    Gives the calibration a known answer (`tests/test_calibration.py` checks the
    fit recovers these parameters) and lets the study and CI run offline.
    """
    rows = []
    for T in maturities:
        strikes = np.array([m * forward for m in moneyness])
        prices = np.atleast_1d(
            heston.call_price(forward, strikes, float(T), 0.0, 0.0, params, n_nodes)
        )
        for K, price in zip(strikes, prices, strict=True):
            iv = bs.implied_vol(float(price), forward, float(K), float(T))
            rows.append(
                {
                    "forward": forward,
                    "strike": float(K),
                    "maturity": float(T),
                    "market_iv": iv,
                    "rel_spread": 0.1,
                    "option_type": "call" if forward <= K else "put",
                    "moneyness": float(K / forward),
                }
            )
    return pd.DataFrame(rows).dropna(subset=["market_iv"]).reset_index(drop=True)
