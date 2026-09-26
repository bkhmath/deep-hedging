"""The benchmark strategies, and the interpolation the Heston delta relies on.

The Heston delta is tabulated on a grid and interpolated, because pricing once per
path and per date would be hundreds of thousands of quadratures. That is a shortcut,
so it is checked against direct calls of the pricer instead of trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from deephedging import black_scholes as bs
from deephedging import heston
from deephedging.benchmarks import bs_delta_positions, heston_delta_positions
from deephedging.heston import HestonParams
from deephedging.simulate import gbm_paths, heston_paths

MILD = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.6)
CRYPTO_LIKE = HestonParams(v0=0.0915, kappa=20.0, theta=0.1761, xi=3.894, rho=-0.107)


def test_bs_delta_positions_uses_the_right_time_to_maturity_at_each_date():
    # The classic off-by-one: the position held over [t_k, t_{k+1}) must be decided
    # with time to maturity T - t_k, not T - t_{k+1}, and there must be exactly n of
    # them for n + 1 observation dates.
    T, K, n = 1.0, 100.0, 20
    S = gbm_paths(50, n, T, 100.0, 0.2, seed=0)
    deltas = bs_delta_positions(S, K, T, 0.2)
    assert deltas.shape == (50, n)
    dt = T / n
    for k in [0, 1, n - 1]:
        expected = bs.call_delta(S[:, k], K, T - dt * k, 0.0, 0.0, 0.2)
        assert np.allclose(deltas[:, k], expected)
    # The last position is decided one step before maturity, so its time to
    # maturity is dt and not zero: a zero would make the delta an indicator.
    assert np.all(deltas[:, -1] < 1.0)


def test_the_interpolated_heston_delta_matches_the_pricer():
    T, K, n = 30.0 / 365.0, 100.0, 30
    S, v = heston_paths(300, n, T, MILD, seed=1, substeps=2)
    interpolated = heston_delta_positions(S, v, K, T, MILD)

    dt = T / n
    rng = np.random.default_rng(0)
    # Spot-check a sample of states against a direct call of the semi-analytic delta.
    errors = []
    for _ in range(60):
        i = int(rng.integers(0, S.shape[0]))
        k = int(rng.integers(0, n))
        exact = float(
            heston.call_delta(
                float(S[i, k]),
                K,
                T - dt * k,
                0.0,
                0.0,
                HestonParams(float(v[i, k]), MILD.kappa, MILD.theta, MILD.xi, MILD.rho),
            )
        )
        errors.append(abs(interpolated[i, k] - exact))
    # Trilinear interpolation on the standardised-moneyness grid of `_heston_grids`.
    # A uniform moneyness grid reached 0.23 here, which is what motivated the
    # change of coordinate; these bounds are the measured values with a little room.
    assert max(errors) < 0.01, max(errors)
    assert float(np.mean(errors)) < 0.002, float(np.mean(errors))


def test_the_heston_delta_lies_in_the_unit_interval():
    T, K = 30.0 / 365.0, 100.0
    for p in [MILD, CRYPTO_LIKE]:
        S, v = heston_paths(2_000, 30, T, p, seed=2, substeps=4)
        deltas = heston_delta_positions(S, v, K, T, p)
        assert deltas.min() > -0.02, deltas.min()
        assert deltas.max() < 1.02, deltas.max()


def test_the_minimum_variance_correction_has_the_sign_of_rho():
    # delta_mv = delta + (rho xi / S) dC/dv, and dC/dv > 0 for a call, so with
    # rho < 0 the minimum-variance hedge must hold strictly fewer shares.
    T, K = 30.0 / 365.0, 100.0
    S, v = heston_paths(2_000, 30, T, MILD, seed=3, substeps=4)
    plain = heston_delta_positions(S, v, K, T, MILD)
    minvar = heston_delta_positions(S, v, K, T, MILD, minimum_variance=True)
    assert MILD.rho < 0.0
    assert np.mean(minvar - plain) < 0.0
    # And with the sign of rho flipped it must go the other way.
    flipped = HestonParams(MILD.v0, MILD.kappa, MILD.theta, MILD.xi, +0.6)
    plain_up = heston_delta_positions(S, v, K, T, flipped)
    minvar_up = heston_delta_positions(S, v, K, T, flipped, minimum_variance=True)
    assert np.mean(minvar_up - plain_up) > 0.0


def test_the_minimum_variance_correction_matches_the_analytic_vega_term():
    # The correction is (rho xi / S) dC/dv with dC/dv from the pricer, so check the
    # tabulated version against a direct finite difference at a handful of states.
    T, K = 30.0 / 365.0, 100.0
    S = np.array([[95.0, 100.0], [105.0, 100.0]])
    v = np.array([[0.04, 0.04], [0.06, 0.06]])
    plain = heston_delta_positions(S, v, K, T, MILD)
    minvar = heston_delta_positions(S, v, K, T, MILD, minimum_variance=True)
    for i in range(2):
        params = HestonParams(float(v[i, 0]), MILD.kappa, MILD.theta, MILD.xi, MILD.rho)
        dC_dv = heston.call_dv(float(S[i, 0]), K, T, 0.0, 0.0, params)
        expected = MILD.rho * MILD.xi / float(S[i, 0]) * dC_dv
        assert minvar[i, 0] - plain[i, 0] == pytest.approx(expected, rel=0.05, abs=0.01)


def test_the_heston_delta_reduces_to_black_scholes_when_the_variance_is_frozen():
    # xi -> 0 with theta = v0 makes the variance deterministic, so the Heston delta
    # evaluated along the paths must agree with the Black-Scholes delta at the
    # corresponding volatility.
    T, K, n = 30.0 / 365.0, 100.0, 20
    frozen = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=1e-3, rho=0.0)
    S, v = heston_paths(500, n, T, frozen, seed=4, substeps=2)
    heston_deltas = heston_delta_positions(S, v, K, T, frozen)
    bs_deltas = bs_delta_positions(S, K, T, np.sqrt(0.04))
    assert np.max(np.abs(heston_deltas - bs_deltas)) < 0.01, np.max(
        np.abs(heston_deltas - bs_deltas)
    )
