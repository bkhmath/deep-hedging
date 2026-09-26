"""The entropic indifference price must tend to the Black-Scholes price as lambda -> 0.

Why this is a real test of the whole pipeline and not of the network: as
lambda -> 0 the entropic measure decreases to the expectation, and

    E[-PnL] = E[payoff] - E[sum_k delta_k (S_{k+1} - S_k)] = E[payoff],

because the paths are simulated under the pricing measure and so the discrete
stochastic integral has mean zero for *any* predictable strategy. The limit is the
risk-neutral expectation of the payoff, which is the Black-Scholes price, and it
does not depend on how well the network hedges. So a failure here indicts the
simulator, the wealth equation, the risk measure or the measure under which the
paths are drawn, not the optimiser.

The flip side, reported in the study and not buried: this is exactly the limit
in which the delta hedge stops being identified. If the objective does not depend
on the strategy, nothing pins the strategy down. The price converges and the
strategy does not, and both statements are consequences of the same expansion.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from deephedging import black_scholes as bs
from deephedging.benchmarks import bs_delta_positions
from deephedging.hedging import pnl
from deephedging.pricing import fit_lambda_intercept, indifference_price
from deephedging.risk import entropic_risk
from deephedging.simulate import gbm_paths
from deephedging.train import TrainConfig, _to_tensor, positions_from_hedger, train_hedger

T, K, S0, SIGMA = 1.0, 100.0, 100.0, 0.2
BS_PRICE = float(bs.call_price(S0, K, T, 0.0, 0.0, SIGMA))


def test_the_delta_hedge_prices_the_call_in_the_zero_risk_aversion_limit():
    # Strategy-independence makes this checkable without training anything.
    S = gbm_paths(200_000, 30, T, S0, SIGMA, seed=3)
    deltas = bs_delta_positions(S, K, T, SIGMA)
    with torch.no_grad():
        loss = -pnl(_to_tensor(S), _to_tensor(deltas), K, 0.0).double()
    se = float(loss.std(unbiased=True) / np.sqrt(len(loss)))
    prices = [float(entropic_risk(loss, lam)) for lam in [0.4, 0.1, 0.02, 0.005]]
    # Decreasing in lambda, down towards the price.
    assert all(a > b for a, b in pairwise(prices)), prices
    assert abs(prices[-1] - BS_PRICE) < 4.0 * se + 0.01, (prices[-1], BS_PRICE, se)


def test_an_arbitrary_strategy_also_prices_the_call_in_the_limit():
    # The limit depends on the martingale property, not on the strategy being good.
    # A poor, path-dependent strategy must give the same price.
    S = gbm_paths(200_000, 30, T, S0, SIGMA, seed=4)
    S_t = _to_tensor(S)
    silly = torch.tanh((S_t[:, :30] - 90.0) / 3.0) * 0.5
    with torch.no_grad():
        loss = -pnl(S_t, silly, K, 0.0).double()
    se = float(loss.std(unbiased=True) / np.sqrt(len(loss)))
    assert abs(float(entropic_risk(loss, 1e-4)) - BS_PRICE) < 4.0 * se, float(
        entropic_risk(loss, 1e-4)
    )


def test_the_second_order_expansion_in_lambda_holds():
    # rho_lambda = E[X] + (lambda/2) Var(X) + O(lambda^2). Confirming the expansion
    # is what licenses reading the intercept of a linear fit as the limit and the
    # slope as half the residual variance.
    S = gbm_paths(100_000, 30, T, S0, SIGMA, seed=5)
    deltas = bs_delta_positions(S, K, T, SIGMA)
    with torch.no_grad():
        loss = -pnl(_to_tensor(S), _to_tensor(deltas), K, 0.0).double()
    mean = float(loss.mean())
    variance = float(loss.var(unbiased=False))
    for lam in [1e-4, 1e-3]:
        predicted = mean + 0.5 * lam * variance
        assert float(entropic_risk(loss, lam)) == pytest.approx(predicted, rel=1e-5)


def test_the_learned_price_extrapolates_to_black_scholes():
    """The full statement, with the network in the loop and a fit in lambda."""
    S_train = gbm_paths(60_000, 30, T, S0, SIGMA, seed=1)
    S_test = gbm_paths(100_000, 30, T, S0, SIGMA, seed=2)
    lambdas = [0.05, 0.1, 0.2, 0.4]
    prices = []
    for lam in lambdas:
        config = TrainConfig(risk="entropic", lam=lam, epochs=25, seed=0)
        result = train_hedger(S_train, K, T, 0.0, config=config)
        learned = positions_from_hedger(result.hedger, S_test)
        prices.append(indifference_price(S_test, learned, K, 0.0, "entropic", lam))

    intercept, slope = fit_lambda_intercept(np.array(lambdas), np.array(prices))
    # Two vol points of the option's value; the residual is dominated by the
    # discretisation of the hedge at 30 dates, not by the limit itself.
    assert abs(intercept - BS_PRICE) < 0.05, (intercept, BS_PRICE)
    # The slope must be positive: a risk-averse seller charges more than the
    # risk-neutral price, and the premium is half the residual variance.
    assert slope > 0.0, slope
    # And the prices must be increasing in lambda.
    assert all(a < b for a, b in pairwise(prices)), prices


def test_the_price_is_above_black_scholes_at_a_positive_risk_aversion():
    # Monotonicity of the entropic measure in lambda, stated as a financial claim:
    # hedging on a discrete grid leaves residual risk, and a risk-averse agent
    # charges for it.
    S = gbm_paths(100_000, 30, T, S0, SIGMA, seed=6)
    deltas = bs_delta_positions(S, K, T, SIGMA)
    price = indifference_price(S, deltas, K, 0.0, "entropic", lam=1.0)
    assert price > BS_PRICE, (price, BS_PRICE)


def test_costs_raise_the_indifference_price():
    # A claim that is more expensive to hedge is worth more to sell.
    S = gbm_paths(100_000, 30, T, S0, SIGMA, seed=7)
    deltas = bs_delta_positions(S, K, T, SIGMA)
    free = indifference_price(S, deltas, K, 0.0, "entropic", lam=1.0)
    costly = indifference_price(S, deltas, K, 0.002, "entropic", lam=1.0)
    assert costly > free
