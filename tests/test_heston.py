"""Heston pricer tests, including the branch cut the little-trap form avoids.

The strongest test in this file is `test_converges_to_black_scholes_at_rate_xi_squared`.
Sending the volatility of variance to zero makes the variance process
deterministic, so the Heston price must converge to the Black-Scholes price: and
it must do so at a specific rate, because the leading correction to the price is
second order in xi (the first-order term vanishes: a symmetric perturbation of the
variance leaves the price unchanged to first order when rho = 0). Checking the
rate and not only the limit means a formula with a wrong coefficient somewhere
cannot pass by being approximately right.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from deephedging import black_scholes as bs
from deephedging import heston
from deephedging.heston import HestonParams

REALISTIC = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7)
# The BTC calibration finds parameters like these: fast mean reversion, large
# volatility of variance, Feller badly violated. Priced here to make sure the
# quadrature copes with the region the study actually uses.
CRYPTO_LIKE = HestonParams(v0=0.0915, kappa=20.0, theta=0.1761, xi=3.894, rho=-0.107)


def test_converges_to_black_scholes_at_rate_xi_squared():
    K = np.array([70.0, 85.0, 100.0, 120.0, 150.0])
    errors = []
    xis = [0.1, 0.05, 0.025]
    for xi in xis:
        p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=xi, rho=0.0)
        err = 0.0
        for T in [0.1, 1.0, 3.0]:
            h = heston.call_price(100.0, K, T, 0.0, 0.0, p, n_nodes=512)
            b = bs.call_price(100.0, K, T, 0.0, 0.0, 0.2)
            err = max(err, float(np.max(np.abs(h - b))))
        errors.append(err)

    # Halving xi must divide the error by about four.
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.0 < ratio < 5.0 for ratio in ratios), (errors, ratios)
    # And the observed order, from a log-log fit, must be close to two.
    order = np.polyfit(np.log(xis), np.log(errors), 1)[0]
    assert order == pytest.approx(2.0, abs=0.15), order


def test_fixed_quadrature_matches_adaptive_quadrature():
    # The pricer uses a fixed Gauss-Legendre rule for speed. Pinning it against
    # scipy's adaptive integration of the same integrand checks that the shortcut
    # costs nothing, at short and long maturity and across the strike range.
    p = REALISTIC
    for T in [0.05, 0.5, 5.0]:
        for K in [70.0, 100.0, 140.0]:

            def integrand(u, shift, K=K, T=T):
                arg = np.array([u - 1j]) if shift else np.array([u])
                phi = heston.char_function(arg, T, 0.0, 0.0, p, S=100.0)[0]
                if shift:
                    phi = phi / 100.0
                return float((np.exp(-1j * u * np.log(K)) * phi / (1j * u)).real)

            P1 = 0.5 + quad(integrand, 1e-12, 500.0, args=(True,), limit=500)[0] / np.pi
            P2 = 0.5 + quad(integrand, 1e-12, 500.0, args=(False,), limit=500)[0] / np.pi
            reference = 100.0 * P1 - K * P2
            # 1e-6 absolute on prices of order 1 to 30, i.e. eight significant
            # figures. The residual is the truncation of the integral at a finite
            # frequency, not the Gauss-Legendre rule, and it is four orders of
            # magnitude below the Monte Carlo error of any hedging number here.
            assert float(heston.call_price(100.0, K, T, 0.0, 0.0, p, 512)) == pytest.approx(
                reference, abs=1e-6
            )


def _char_function_original_branch(u, T, p, S=1.0):
    """Heston's 1993 form, with g1 and exp(+dT). Present only to be tested against.

    This is the formulation the "little trap" paper warns about. It is
    algebraically identical to `heston.char_function` and numerically different,
    because |g1| > 1 makes the argument of the complex logarithm wind around the
    origin as T grows, and the principal branch then jumps.
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi**2 * (1j * u + u**2))
    plus = kappa - rho * xi * 1j * u + d
    g = plus / (kappa - rho * xi * 1j * u - d)
    exp_dT = np.exp(d * T)
    C = (kappa * theta / xi**2) * (plus * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
    D = (v0 / xi**2) * plus * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(1j * u * np.log(S) + C + D)


def test_the_little_trap_and_the_original_form_agree_only_at_short_maturity():
    # This is the whole point of the module docstring, as a measurement. The two
    # formulations are the same function on paper. At half a year they agree to
    # machine precision; by three years the original form is wrong by an amount
    # of order one, which is the branch-cut crossing.
    p = REALISTIC
    u = np.linspace(0.01, 30.0, 40)
    short = np.max(
        np.abs(
            heston.char_function(u, 0.5, 0.0, 0.0, p) - _char_function_original_branch(u, 0.5, p)
        )
    )
    long = np.max(
        np.abs(
            heston.char_function(u, 3.0, 0.0, 0.0, p) - _char_function_original_branch(u, 3.0, p)
        )
    )
    assert short < 1e-12, short
    assert long > 0.1, long


def _price_from_char_function(cf, K, T, p, n_nodes=512):
    """Price with an arbitrary characteristic function, to price the original form."""
    u, w = heston._gauss_legendre_nodes(n_nodes, 200.0)
    phi2 = cf(u, T, p, 100.0)
    phi1 = cf(u - 1j, T, p, 100.0) / 100.0
    phase = np.exp(-1j * u * np.log(K))
    P1 = 0.5 + np.real(phase * phi1 / (1j * u)) @ w / np.pi
    P2 = 0.5 + np.real(phase * phi2 / (1j * u)) @ w / np.pi
    return 100.0 * P1 - K * P2


def test_the_little_trap_form_is_smooth_in_maturity_where_the_original_is_not():
    # A branch-cut crossing shows up as a kink in the price as a function of T.
    # The absolute size of a third difference is not meaningful on its own: the
    # price has genuine curvature in T, so the two formulations are compared
    # against each other on the same grid. The original form is worse by orders of
    # magnitude, and that gap is the bug the little trap removes.
    p = REALISTIC
    maturities = np.linspace(0.25, 5.0, 200)
    trap = np.array(
        [float(heston.call_price(100.0, 100.0, float(T), 0.0, 0.0, p)) for T in maturities]
    )
    original = np.array(
        [
            _price_from_char_function(_char_function_original_branch, 100.0, float(T), p)
            for T in maturities
        ]
    )
    trap_roughness = float(np.max(np.abs(np.diff(trap, n=3))))
    original_roughness = float(np.max(np.abs(np.diff(original, n=3))))
    assert original_roughness > 100.0 * trap_roughness, (trap_roughness, original_roughness)
    # And the trap form stays monotone in maturity, as a call must be.
    assert np.all(np.diff(trap) > 0.0)


def test_delta_matches_a_central_finite_difference():
    # The delta is claimed to equal e^{-qT} P1 by a homogeneity argument. This
    # checks the argument numerically instead of trusting the derivation.
    for p in [REALISTIC, CRYPTO_LIKE]:
        for T in [0.08, 1.0]:
            for K in [80.0, 100.0, 130.0]:
                h = 1e-3
                fd = (
                    float(heston.call_price(100.0 + h, K, T, 0.0, 0.0, p, 512))
                    - float(heston.call_price(100.0 - h, K, T, 0.0, 0.0, p, 512))
                ) / (2 * h)
                exact = float(heston.call_delta(100.0, K, T, 0.0, 0.0, p, 512))
                assert exact == pytest.approx(fd, abs=2e-5), (p, T, K, exact, fd)


def test_dv_matches_a_finite_difference_in_the_initial_variance():
    p = REALISTIC
    direct = heston.call_dv(100.0, 100.0, 1.0, 0.0, 0.0, p, h=1e-5)
    coarse = heston.call_dv(100.0, 100.0, 1.0, 0.0, 0.0, p, h=1e-4)
    assert direct == pytest.approx(coarse, rel=1e-4)
    # Vega is positive: more variance is worth more to the holder of a call.
    assert direct > 0.0


def test_price_respects_no_arbitrage_and_monotonicity_in_strike():
    for p in [REALISTIC, CRYPTO_LIKE]:
        K = np.linspace(40.0, 300.0, 80)
        price = heston.call_price(100.0, K, 0.75, 0.0, 0.0, p)
        assert np.all(np.diff(price) <= 1e-10)  # decreasing in strike
        assert np.all(price >= np.maximum(100.0 - K, 0.0) - 1e-10)
        assert np.all(price <= 100.0 + 1e-10)
        # Convexity in strike, which is the butterfly no-arbitrage condition.
        assert np.all(np.diff(price, n=2) >= -1e-7)


def test_feller_reports_the_sign_correctly():
    assert HestonParams(0.04, 2.0, 0.04, 0.2, -0.5).feller() > 0
    assert CRYPTO_LIKE.feller() < 0
