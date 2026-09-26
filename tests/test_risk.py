"""Risk measure tests: the defining properties, and the numerical stability claim.

The entropic measure and CVaR are both claimed to be convex monetary risk measures,
so both must be translation invariant, monotone and convex. Those are checkable
statements and they are checked here instead of assumed, because a sign error in
either would silently change what the network is optimising while leaving it
apparently training.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from deephedging.risk import cvar, entropic_risk, evaluate_risk, make_objective


def sample(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.standard_normal(n) * 2.0 - 0.5, dtype=torch.float64)


def test_entropic_risk_does_not_overflow_on_losses_that_would_break_a_naive_exponential():
    # exp(700) is the last finite double, and exp(89) the last finite float32. A
    # single loss of 1000 is therefore enough to turn a naive implementation into
    # inf, and its gradient into NaN. This is not contrived: at initialisation the
    # network holds no position, so the loss is the raw option payoff, and in the
    # crypto-calibrated experiments that reaches the hundreds.
    loss = torch.tensor([1000.0, 0.0, -5.0], dtype=torch.float64)
    value = entropic_risk(loss, lam=1.0)
    assert torch.isfinite(value)
    # With lambda = 1 the worst loss dominates: the measure is the maximum plus
    # log(1/3) of the count, to floating point.
    assert float(value) == pytest.approx(1000.0 + np.log(1.0 / 3.0), abs=1e-9)

    # And in float32, which is what training actually uses.
    value32 = entropic_risk(loss.float(), lam=1.0)
    assert torch.isfinite(value32)


def test_entropic_risk_gradient_is_finite_for_extreme_losses():
    x = torch.tensor([500.0, 1.0, -2.0], dtype=torch.float32, requires_grad=True)
    entropic_risk(x, lam=1.0).backward()
    assert torch.all(torch.isfinite(x.grad))
    # The weight sits almost entirely on the worst outcome, as exponential utility says.
    assert float(x.grad[0]) == pytest.approx(1.0, abs=1e-6)


def test_entropic_risk_matches_the_direct_formula_where_the_direct_formula_works():
    loss = sample()
    for lam in [0.1, 0.5, 2.0]:
        direct = torch.log(torch.mean(torch.exp(lam * loss))) / lam
        assert float(entropic_risk(loss, lam)) == pytest.approx(float(direct), rel=1e-10)


def test_entropic_risk_is_translation_invariant():
    loss = sample()
    for lam in [0.2, 1.0, 3.0]:
        base = float(entropic_risk(loss, lam))
        shifted = float(entropic_risk(loss + 7.0, lam))
        assert shifted == pytest.approx(base + 7.0, abs=1e-9)


def test_entropic_risk_increases_with_risk_aversion_and_tends_to_the_mean():
    loss = sample()
    values = [float(entropic_risk(loss, lam)) for lam in [0.001, 0.01, 0.1, 1.0, 2.0]]
    assert all(a < b for a, b in pairwise(values))
    # As lambda -> 0 it decreases to the expectation. This limit is what makes the
    # indifference price converge to the Black-Scholes price.
    assert values[0] == pytest.approx(float(loss.mean()), abs=2e-3)
    # And the second-order expansion rho = E + (lambda/2) Var holds for small lambda.
    lam = 1e-3
    predicted = float(loss.mean()) + 0.5 * lam * float(loss.var(unbiased=False))
    assert float(entropic_risk(loss, lam)) == pytest.approx(predicted, rel=1e-4)


def test_entropic_risk_is_convex():
    # rho(t X + (1-t) Y) <= t rho(X) + (1-t) rho(Y) for the same underlying states.
    rng = np.random.default_rng(1)
    X = torch.as_tensor(rng.standard_normal(5000), dtype=torch.float64)
    Y = torch.as_tensor(rng.standard_normal(5000) * 3.0, dtype=torch.float64)
    for t in [0.25, 0.5, 0.75]:
        mixed = float(entropic_risk(t * X + (1 - t) * Y, 1.0))
        bound = t * float(entropic_risk(X, 1.0)) + (1 - t) * float(entropic_risk(Y, 1.0))
        assert mixed <= bound + 1e-9


def test_entropic_risk_rejects_a_non_positive_lambda():
    with pytest.raises(ValueError):
        entropic_risk(sample(), lam=0.0)


def test_cvar_equals_the_mean_of_the_worst_tail():
    # The Rockafellar-Uryasev form must reproduce the definition: the average of
    # the losses beyond the alpha quantile.
    loss = sample(n=100_000, seed=2)
    for alpha in [0.9, 0.95, 0.99]:
        q = torch.quantile(loss, alpha)
        tail_mean = float(loss[loss >= q].mean())
        assert float(cvar(loss, alpha)) == pytest.approx(tail_mean, rel=2e-3)


def test_cvar_of_a_normal_loss_matches_the_closed_form():
    # For X ~ N(mu, sigma^2), CVaR_alpha = mu + sigma phi(z_alpha) / (1 - alpha).
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    mu, sd, alpha = 1.5, 2.0, 0.95
    loss = torch.as_tensor(mu + sd * rng.standard_normal(500_000), dtype=torch.float64)
    closed = mu + sd * norm.pdf(norm.ppf(alpha)) / (1.0 - alpha)
    assert float(cvar(loss, alpha)) == pytest.approx(closed, rel=5e-3)


def test_the_rockafellar_uryasev_objective_is_minimised_at_the_value_at_risk():
    # The point of the representation: w + E[(X-w)^+]/(1-alpha) is minimised at the
    # alpha quantile, and its minimum is the CVaR. Both halves are checked.
    loss = sample(n=50_000, seed=4)
    alpha = 0.95
    var = float(torch.quantile(loss, alpha))
    at_optimum = float(cvar(loss, alpha, w=torch.tensor(var, dtype=torch.float64)))
    assert at_optimum == pytest.approx(float(cvar(loss, alpha)), rel=1e-9)
    for offset in [-1.0, -0.2, 0.2, 1.0]:
        w = torch.tensor(var + offset, dtype=torch.float64)
        assert float(cvar(loss, alpha, w=w)) >= at_optimum - 1e-9


def test_cvar_is_translation_invariant_and_at_least_the_mean():
    loss = sample()
    assert float(cvar(loss + 3.0, 0.95)) == pytest.approx(float(cvar(loss, 0.95)) + 3.0, abs=1e-9)
    assert float(cvar(loss, 0.95)) > float(loss.mean())


def test_cvar_rejects_a_level_outside_the_open_unit_interval():
    for alpha in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            cvar(sample(), alpha)


def test_make_objective_exposes_the_cvar_auxiliary_variable_and_nothing_else():
    objective, extra = make_objective("entropic", lam=1.0)
    assert extra == []
    objective, extra = make_objective("cvar", alpha=0.95)
    assert len(extra) == 1 and extra[0].requires_grad
    # The auxiliary variable must actually receive a gradient, otherwise it would
    # never move and the CVaR would be evaluated at a fixed, wrong threshold.
    objective(sample().float()).backward()
    assert extra[0].grad is not None and torch.all(torch.isfinite(extra[0].grad))


def test_unknown_risk_measure_is_rejected():
    with pytest.raises(ValueError):
        make_objective("variance")
    with pytest.raises(ValueError):
        evaluate_risk(sample(), "variance")
