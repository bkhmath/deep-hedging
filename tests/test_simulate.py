"""Simulator tests, including the demonstration that the naive variance scheme fails.

The most important test here is `test_heston_monte_carlo_price_matches_the_pricer`:
it is the only thing that ties the simulator and the semi-analytic pricer together.
They are independent implementations of the same model, one a discretisation of
the SDE, the other a contour integral of a characteristic function, so if they
agree on a price to within the Monte Carlo error, both are almost certainly right.
Neither one alone would establish that.
"""

from __future__ import annotations

import numpy as np
import pytest

from deephedging import heston
from deephedging.heston import HestonParams
from deephedging.simulate import call_payoff, gbm_paths, heston_paths

# Calibrated to the BTC surface: fast mean reversion, huge volatility of variance,
# Feller violated. Deliberately the hard case.
CRYPTO_LIKE = HestonParams(v0=0.0915, kappa=20.0, theta=0.1761, xi=3.894, rho=-0.107)
MILD = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.6)


def test_gbm_terminal_law_is_lognormal_with_the_right_moments():
    T, S0, sigma = 1.0, 100.0, 0.25
    S = gbm_paths(400_000, 50, T, S0, sigma, seed=3)
    log_return = np.log(S[:, -1] / S0)
    # Under the pricing measure with r = q = 0 the log return has mean
    # -sigma^2 T / 2 and variance sigma^2 T.
    assert log_return.mean() == pytest.approx(-0.5 * sigma**2 * T, abs=4e-3)
    assert log_return.std(ddof=1) == pytest.approx(sigma * np.sqrt(T), rel=5e-3)


def test_gbm_is_a_martingale_under_the_pricing_measure():
    S = gbm_paths(400_000, 30, 1.0, 100.0, 0.3, seed=4)
    for k in [1, 10, 30]:
        se = S[:, k].std(ddof=1) / np.sqrt(S.shape[0])
        assert abs(S[:, k].mean() - 100.0) < 4.0 * se


def test_gbm_price_matches_black_scholes():
    from deephedging import black_scholes as bs

    T, K, sigma = 0.5, 105.0, 0.2
    S = gbm_paths(400_000, 20, T, 100.0, sigma, seed=5)
    payoff = call_payoff(S[:, -1], K)
    se = payoff.std(ddof=1) / np.sqrt(len(payoff))
    reference = float(bs.call_price(100.0, K, T, 0.0, 0.0, sigma))
    assert abs(payoff.mean() - reference) < 4.0 * se


def test_gbm_is_reproducible_and_seed_dependent():
    a = gbm_paths(1000, 10, 1.0, 100.0, 0.2, seed=11)
    b = gbm_paths(1000, 10, 1.0, 100.0, 0.2, seed=11)
    c = gbm_paths(1000, 10, 1.0, 100.0, 0.2, seed=12)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_naive_euler_variance_scheme_produces_nan():
    """The failure that full truncation exists to prevent, measured.

    This reimplements the textbook Euler step with no truncation and shows that on
    the parameters this project actually calibrates to it breaks almost at once:
    the variance goes negative, sqrt of it is NaN, and the NaN then propagates
    through every subsequent step. It is not a rare tail event, it is the normal
    behaviour when the Feller condition is violated.
    """
    assert CRYPTO_LIKE.feller() < 0
    rng = np.random.default_rng(0)
    n_paths, n_fine, T = 20_000, 120, 30.0 / 365.0
    dt = T / n_fine
    v = np.full(n_paths, CRYPTO_LIKE.v0)
    first_nan = None
    with np.errstate(invalid="ignore"):
        for i in range(n_fine):
            z = rng.standard_normal(n_paths)
            v = (
                v
                + CRYPTO_LIKE.kappa * (CRYPTO_LIKE.theta - v) * dt
                + CRYPTO_LIKE.xi * np.sqrt(v) * np.sqrt(dt) * z
            )
            if first_nan is None and np.isnan(v).any():
                first_nan = i + 1
    assert first_nan is not None and first_nan < 10, first_nan
    assert np.isnan(v).mean() > 0.5, np.isnan(v).mean()


def test_full_truncation_never_produces_nan_or_negative_variance():
    S, v = heston_paths(50_000, 30, 30.0 / 365.0, CRYPTO_LIKE, seed=1, substeps=4)
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(v))
    assert np.all(v >= 0.0)
    assert np.all(S > 0.0)
    # And the zero boundary is reached, so the scheme is being exercised
    # instead of merely tolerated: with Feller violated the variance touches zero
    # on a substantial fraction of observations.
    assert (v[:, 1:] == 0.0).mean() > 0.01


def test_full_truncation_leaves_the_zero_boundary_non_absorbing():
    # A path that hits zero must be able to recover, because the true process does.
    # Absorption at zero would be a different model.
    _, v = heston_paths(20_000, 30, 30.0 / 365.0, CRYPTO_LIKE, seed=2, substeps=4)
    hit_zero = (v[:, 1:-1] == 0.0).any(axis=1)
    assert hit_zero.sum() > 100
    recovered = (v[hit_zero, -1] > 0.0).mean()
    assert recovered > 0.5, recovered


def test_heston_variance_mean_follows_the_ornstein_uhlenbeck_law():
    # E[v_t] = theta + (v0 - theta) exp(-kappa t), exactly, for the true process.
    # The scheme must reproduce it within Monte Carlo error.
    p = MILD
    T, n = 1.0, 50
    _, v = heston_paths(200_000, n, T, p, seed=6, substeps=4)
    times = np.linspace(0.0, T, n + 1)
    expected = p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * times)
    se = v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
    z = (v.mean(axis=0) - expected) / np.maximum(se, 1e-12)
    assert np.max(np.abs(z[1:])) < 5.0, np.max(np.abs(z[1:]))


def test_heston_is_a_martingale_under_the_pricing_measure():
    S, _ = heston_paths(200_000, 30, 0.5, MILD, seed=7, substeps=4)
    se = S[:, -1].std(ddof=1) / np.sqrt(S.shape[0])
    assert abs(S[:, -1].mean() - 100.0) < 4.0 * se


@pytest.mark.parametrize("p", [MILD, CRYPTO_LIKE])
def test_heston_monte_carlo_price_matches_the_pricer(p):
    """The simulator and the semi-analytic pricer, independently, on one price."""
    T, K = 30.0 / 365.0, 100.0
    S, _ = heston_paths(200_000, 30, T, p, seed=8, substeps=4)
    payoff = call_payoff(S[:, -1], K)
    se = payoff.std(ddof=1) / np.sqrt(len(payoff))
    reference = float(heston.call_price(100.0, K, T, 0.0, 0.0, p))
    assert abs(payoff.mean() - reference) < 4.0 * se, (payoff.mean(), reference, se)


def test_substeps_refine_the_scheme_without_changing_the_model():
    # Refining the simulation grid must not move the price outside Monte Carlo
    # error: it is the same model, more accurately discretised.
    T, K = 30.0 / 365.0, 100.0
    reference = float(heston.call_price(100.0, K, T, 0.0, 0.0, CRYPTO_LIKE))
    for substeps in [1, 4, 16]:
        S, _ = heston_paths(100_000, 30, T, CRYPTO_LIKE, seed=9, substeps=substeps)
        payoff = call_payoff(S[:, -1], K)
        se = payoff.std(ddof=1) / np.sqrt(len(payoff))
        assert abs(payoff.mean() - reference) < 4.0 * se, substeps


def test_substeps_reduce_how_often_the_variance_sticks_at_zero():
    # Touching zero is partly a property of the model and partly an artefact of a
    # coarse step. Refining must reduce the artefact, monotonically.
    fractions = []
    for substeps in [1, 4, 16]:
        _, v = heston_paths(20_000, 30, 30.0 / 365.0, CRYPTO_LIKE, seed=10, substeps=substeps)
        fractions.append(float((v[:, 1:] == 0.0).mean()))
    assert fractions[0] > fractions[1] > fractions[2], fractions


def test_shapes_and_initial_conditions():
    S, v = heston_paths(17, 13, 0.4, MILD, S0=123.0, seed=0, substeps=3)
    assert S.shape == (17, 14) and v.shape == (17, 14)
    # The log price is stepped and not the price, so the initial spot comes
    # back as exp(log(123.0)), which is 123.0 only to rounding.
    assert S[:, 0] == pytest.approx(123.0)
    assert np.all(v[:, 0] == MILD.v0)
    with pytest.raises(ValueError):
        heston_paths(10, 5, 0.5, MILD, substeps=0)
