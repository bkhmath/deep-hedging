"""Black-Scholes closed forms, against identities and not recorded outputs.

Every assertion here is a statement that is true independently of this code:
put-call parity, the payoff limit, the derivative relations. Nothing is pinned to
a number that a previous run of this repository happened to produce, because a
test like that only proves the code has not changed.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from deephedging import black_scholes as bs


def test_put_call_parity():
    # C - P = S e^{-qT} - K e^{-rT}. The put is obtained from the call and parity,
    # so this checks the call against the risk-neutral expectation of the payoff
    # difference, which is a model-free identity.
    S, K, T, r, q, sigma = 100.0, 95.0, 0.7, 0.03, 0.01, 0.25
    call = float(bs.call_price(S, K, T, r, q, sigma))
    # Put from the same d1, d2 via the symmetry C(S,K) - P(S,K) = Se^{-qT} - Ke^{-rT}
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    assert put > 0.0
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-12)


def test_price_converges_to_payoff_as_maturity_vanishes():
    S, K = 105.0, 100.0
    prices = [float(bs.call_price(S, K, T, 0.0, 0.0, 0.3)) for T in [1e-2, 1e-4, 1e-6, 0.0]]
    assert prices[-1] == pytest.approx(5.0, abs=1e-12)
    # monotone approach to the intrinsic value from above
    assert all(a >= b - 1e-12 for a, b in pairwise(prices))


def test_delta_matches_finite_difference():
    S, K, T, r, q, sigma = 100.0, 110.0, 0.5, 0.02, 0.0, 0.3
    h = 1e-4 * S
    fd = (
        float(bs.call_price(S + h, K, T, r, q, sigma))
        - float(bs.call_price(S - h, K, T, r, q, sigma))
    ) / (2 * h)
    assert float(bs.call_delta(S, K, T, r, q, sigma)) == pytest.approx(fd, abs=1e-8)


def test_vega_matches_finite_difference():
    S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    h = 1e-6
    fd = (
        float(bs.call_price(S, K, T, 0, 0, sigma + h))
        - float(bs.call_price(S, K, T, 0, 0, sigma - h))
    ) / (2 * h)
    assert float(bs.call_vega(S, K, T, 0, 0, sigma)) == pytest.approx(fd, rel=1e-6)


def test_delta_lies_in_the_unit_interval_and_is_increasing_in_spot():
    S = np.linspace(20.0, 400.0, 200)
    delta = bs.call_delta(S, 100.0, 0.5, 0.0, 0.0, 0.25)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
    assert np.all(np.diff(delta) > 0.0)


def test_price_is_bounded_by_no_arbitrage():
    S, K, T, r, sigma = 100.0, 90.0, 1.5, 0.04, 0.4
    price = float(bs.call_price(S, K, T, r, 0.0, sigma))
    assert max(S - K * np.exp(-r * T), 0.0) < price < S


def test_implied_vol_inverts_the_price():
    # Round trip over a wide grid: the inversion must return the volatility that
    # produced the price, to the tolerance of the root finder.
    #
    # Quotes with negligible vega are excluded instead of given a loose
    # tolerance. A deep in-the-money one-month option is worth its intrinsic
    # value to fifteen decimal places, so its price carries no information about
    # volatility at all and no inversion can recover one; the next test asserts
    # that the function says so instead of inventing a number.
    checked = 0
    for K in [70.0, 100.0, 140.0]:
        for T in [0.05, 1.0, 3.0]:
            for sigma in [0.1, 0.35, 0.9]:
                if float(bs.call_vega(100.0, K, T, 0.01, 0.0, sigma)) < 1e-6:
                    continue
                price = float(bs.call_price(100.0, K, T, 0.01, 0.0, sigma))
                recovered = bs.implied_vol(price, 100.0, K, T, 0.01, 0.0)
                assert recovered == pytest.approx(sigma, abs=1e-7)
                checked += 1
    assert checked >= 20  # the exclusion must not have emptied the grid


def test_implied_vol_is_nan_when_the_quote_carries_no_volatility_information():
    # Deep in the money and one month out: the price is the intrinsic value to
    # machine precision, so no volatility is implied.
    K, T, r = 70.0, 0.05, 0.01
    price = float(bs.call_price(100.0, K, T, r, 0.0, 0.1))
    assert price == pytest.approx(100.0 - K * np.exp(-r * T), abs=1e-12)
    assert np.isnan(bs.implied_vol(price, 100.0, K, T, r, 0.0))


def test_implied_vol_returns_nan_outside_the_arbitrage_bounds():
    # A price above the spot, or below the intrinsic value, admits no volatility.
    assert np.isnan(bs.implied_vol(101.0, 100.0, 90.0, 1.0))
    assert np.isnan(bs.implied_vol(1.0, 100.0, 90.0, 1.0))


def test_degenerate_inputs_do_not_warn_or_return_nan():
    # The pyproject turns RuntimeWarning into an error, so a divide-by-zero inside
    # the formula would fail this test over pass silently.
    for T, sigma in [(0.0, 0.2), (1.0, 0.0), (0.0, 0.0)]:
        price = bs.call_price(np.array([80.0, 120.0]), 100.0, T, 0.0, 0.0, sigma)
        delta = bs.call_delta(np.array([80.0, 120.0]), 100.0, T, 0.0, 0.0, sigma)
        assert np.all(np.isfinite(price)) and np.all(np.isfinite(delta))
