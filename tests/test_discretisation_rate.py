"""The delta hedge's replication error must decay like n^{-1/2}, and the rate is checked.

Hedging a call on a grid of n dates instead of continuously leaves an error whose
standard deviation is of order n^{-1/2}. The mechanism: over a step of length dt the
delta hedge misses the second-order term of the profit and loss, which is
proportional to the gamma times (dS)^2 - E[(dS)^2]; those terms are uncorrelated
across steps, so their sum has variance of order n * dt^2 = T^2 / n, hence a
standard deviation of order n^{-1/2}. (The classical reference is Bertsimas, Kogan
and Lo, 2000.)

Checking the rate and not only that the error decreases is the point. A wrong time
grid, an off-by-one in which delta is used on which interval, or a missing terminal
liquidation would all leave the error decreasing: just at the wrong exponent, or
towards a non-zero floor.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from deephedging import black_scholes as bs
from deephedging.benchmarks import bs_delta_positions
from deephedging.hedging import pnl, turnover
from deephedging.simulate import gbm_paths
from deephedging.train import _to_tensor

T, K, S0, SIGMA = 1.0, 100.0, 100.0, 0.2
BS_PRICE = float(bs.call_price(S0, K, T, 0.0, 0.0, SIGMA))


def replication_error(n_steps: int, n_paths: int = 100_000, seed: int = 0) -> np.ndarray:
    """Terminal wealth of the delta hedge plus the premium received at inception."""
    S = gbm_paths(n_paths, n_steps, T, S0, SIGMA, seed=seed)
    deltas = bs_delta_positions(S, K, T, SIGMA)
    with torch.no_grad():
        terminal = pnl(_to_tensor(S), _to_tensor(deltas), K, 0.0).double().numpy()
    return terminal + BS_PRICE


def test_the_error_decays_at_the_square_root_rate():
    ns = [8, 16, 32, 64, 128]
    stds = [float(replication_error(n, seed=100 + n).std(ddof=1)) for n in ns]
    assert all(a > b for a, b in pairwise(stds)), stds
    slope = float(np.polyfit(np.log(ns), np.log(stds), 1)[0])
    assert abs(slope + 0.5) < 0.05, slope
    # Each doubling of n must divide the error by close to sqrt(2).
    ratios = [stds[i] / stds[i + 1] for i in range(len(stds) - 1)]
    assert all(1.30 < r < 1.50 for r in ratios), ratios


def test_the_delta_hedge_is_unbiased_at_every_refinement():
    # The hedging gain is a sum of martingale increments, so the mean replication
    # error is zero for any n. A non-zero mean would mean the premium, the payoff or
    # the gain term is mis-stated: a bias that no amount of refinement removes.
    for n in [8, 32, 128]:
        error = replication_error(n, seed=200 + n)
        se = error.std(ddof=1) / np.sqrt(len(error))
        assert abs(error.mean()) < 4.0 * se, (n, error.mean(), se)


def test_the_error_is_not_converging_to_a_non_zero_floor():
    # A floor would show up as the ratio of successive standard deviations dropping
    # towards one at large n. It must not.
    fine = [float(replication_error(n, seed=300 + n).std(ddof=1)) for n in [128, 256]]
    assert fine[0] / fine[1] > 1.30, fine


def test_transaction_costs_destroy_the_benefit_of_rebalancing_more_often():
    """The finding that motivates the whole project, as a test.

    Frictionless, refining the hedging grid always helps. With proportional costs it
    stops helping: each rebalance trades a notional of order |Gamma| * |dS| * S,
    which is of order sqrt(dt), and there are n of them, so the total traded
    notional grows like n * sqrt(T/n) = sqrt(nT). Against a replication error that
    only falls like n^{-1/2}, the two meet at a finite optimal frequency and beyond
    it more hedging is strictly worse.

    The measured turnover is fitted to A + B*sqrt(n) and not to a pure power law,
    because there is an unavoidable additive piece: whatever n is, the hedge must buy
    the initial delta once and liquidate it once. For an at-the-money call that is
    about 0.5 shares each way at a spot of 100, so A should come out near 100: and
    it does, which is a sharper statement than a fitted exponent would be.
    """
    ns = np.array([8, 16, 32, 64, 128, 256, 512])
    turnovers = []
    for n in ns:
        S = gbm_paths(50_000, int(n), T, S0, SIGMA, seed=400 + int(n))
        deltas = bs_delta_positions(S, K, T, SIGMA)
        with torch.no_grad():
            turnovers.append(float(turnover(_to_tensor(S), _to_tensor(deltas)).mean()))
    turnovers = np.array(turnovers)

    # Turnover grows monotonically with the rebalancing frequency.
    assert np.all(np.diff(turnovers) > 0.0), turnovers

    # Fit A + B sqrt(n) and check it explains essentially all of the variation.
    design = np.stack([np.ones_like(ns, dtype=float), np.sqrt(ns)], axis=1)
    (A, B), *_ = np.linalg.lstsq(design, turnovers, rcond=None)
    fitted = design @ np.array([A, B])
    r_squared = 1.0 - np.sum((turnovers - fitted) ** 2) / np.sum(
        (turnovers - turnovers.mean()) ** 2
    )
    assert r_squared > 0.9995, r_squared
    assert B > 0.0, B
    # The additive piece is the unavoidable round trip of the initial delta.
    initial_delta = float(bs.call_delta(S0, K, T, 0.0, 0.0, SIGMA))
    assert pytest.approx(2.0 * initial_delta * S0, rel=0.25) == A, (A, initial_delta)

    # The pure log-log slope therefore approaches 1/2 from below as the fitting
    # window moves to larger n, and must not exceed it.
    slopes = [float(np.polyfit(np.log(ns[i:]), np.log(turnovers[i:]), 1)[0]) for i in [0, 2, 3]]
    assert all(a < b for a, b in pairwise(slopes)), slopes
    assert 0.3 < slopes[-1] < 0.5, slopes

    # And the expected cost paid grows with n at a positive rate, so that hedging
    # more often is strictly more expensive.
    cost_rate = 0.002
    assert cost_rate * turnovers[-1] > 2.0 * cost_rate * turnovers[0]


def _asymptotic_error_std(n: int) -> float:
    """The limiting standard deviation of the delta hedge's replication error.

    Over one step the delta hedge misses the second-order term of the profit and
    loss. With r = 0 the Black-Scholes equation gives theta = -(1/2) sigma^2 S^2
    Gamma, so the increment is

        (1/2) Gamma [ (dS)^2 - sigma^2 S^2 dt ]  =  (1/2) sigma^2 S^2 Gamma dt (Z^2 - 1)

    to leading order, with Z standard normal. Var(Z^2 - 1) = 2, and the increments
    are uncorrelated across steps, so

        Var(error) -> (1/2) (T/n) * integral_0^T E[(sigma^2 S_t^2 Gamma_t)^2] dt.

    The integrand has an integrable (T - t)^{-1/2} singularity at maturity, because
    the gamma of a call blows up there. The substitution t = T - s^2 removes it
    exactly, which is why the quadrature below is in s and not in t.
    """
    from scipy.stats import norm

    def expectation(t: float) -> float:
        tau = T - t
        z = np.linspace(-8.0, 8.0, 2001)
        S = S0 * np.exp(-0.5 * SIGMA**2 * t + SIGMA * np.sqrt(t) * z)
        d1 = (np.log(S / K) + 0.5 * SIGMA**2 * tau) / (SIGMA * np.sqrt(tau))
        gamma = norm.pdf(d1) / (S * SIGMA * np.sqrt(tau))
        return float(np.trapezoid((SIGMA**2 * S**2 * gamma) ** 2 * norm.pdf(z), z))

    # t = T - s^2, dt = -2 s ds, s from 0 (maturity) to sqrt(T) (inception).
    nodes, weights = np.polynomial.legendre.leggauss(120)
    s = 0.5 * np.sqrt(T) * (nodes + 1.0)
    w = 0.5 * np.sqrt(T) * weights
    integral = float(np.sum(w * 2.0 * s * np.array([expectation(T - si**2) for si in s])))
    return float(np.sqrt(0.5 * integral * T / n))


def test_the_error_standard_deviation_matches_its_asymptotic_constant():
    """The constant, not just the exponent.

    Matching an asymptotic constant leaves very little room for a pair of
    compensating errors: a mis-specified time grid would change the constant even
    where it preserved the rate. The agreement must also *improve* with n, since the
    prediction is a limit, so the test checks the trend as well as the level.
    """
    ratios = []
    for n in [64, 128, 256]:
        measured = float(replication_error(n, n_paths=200_000, seed=7).std(ddof=1))
        ratios.append(measured / _asymptotic_error_std(n))
    # Within a few per cent, and approaching one from below as n grows.
    assert all(0.95 < r < 1.01 for r in ratios), ratios
    assert all(a < b for a, b in pairwise(ratios)), ratios
