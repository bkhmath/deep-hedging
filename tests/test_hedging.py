"""The wealth equation, checked against hand-computable cases.

The profit and loss function is the one place where a sign error would corrupt
every number in the study while leaving everything apparently working, so it is
tested on cases small enough to evaluate on paper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedging.hedging import hedging_gain, pnl, trade_sizes, transaction_cost, turnover


def test_trade_sizes_start_flat_and_liquidate():
    deltas = torch.tensor([[0.3, 0.5, 0.2]])
    trades = trade_sizes(deltas)
    # delta_0 - 0, delta_1 - delta_0, delta_2 - delta_1, 0 - delta_2
    assert trades.shape == (1, 4)
    assert trades[0].tolist() == pytest.approx([0.3, 0.2, -0.3, -0.2])
    # The trades must sum to zero: the position starts flat and ends flat.
    assert float(trades.sum()) == pytest.approx(0.0, abs=1e-7)


def test_no_hedge_loses_exactly_the_payoff():
    S = torch.tensor([[100.0, 110.0, 130.0]])
    deltas = torch.zeros(1, 2)
    assert float(pnl(S, deltas, strike=100.0, cost_rate=0.0)) == pytest.approx(-30.0)
    # And nothing is traded, so nothing is charged even at a punitive cost rate.
    assert float(pnl(S, deltas, 100.0, cost_rate=0.5)) == pytest.approx(-30.0)


def test_a_hand_computed_two_step_path():
    # S: 100 -> 110 -> 90, strike 100, positions 0.5 then 0.8, cost 1%.
    #   payoff        = max(90 - 100, 0) = 0
    #   hedging gain  = 0.5 * (110 - 100) + 0.8 * (90 - 110) = 5 - 16 = -11
    #   trades        = +0.5 at 100, +0.3 at 110, -0.8 at 90
    #   cost          = 0.01 * (0.5*100 + 0.3*110 + 0.8*90) = 0.01 * (50 + 33 + 72) = 1.55
    S = torch.tensor([[100.0, 110.0, 90.0]])
    deltas = torch.tensor([[0.5, 0.8]])
    assert float(hedging_gain(S, deltas)) == pytest.approx(-11.0)
    assert float(transaction_cost(S, deltas, 0.01)) == pytest.approx(1.55)
    assert float(turnover(S, deltas)) == pytest.approx(155.0)
    assert float(pnl(S, deltas, 100.0, 0.01)) == pytest.approx(-12.55)


def test_the_terminal_liquidation_is_charged():
    # A position held into maturity and never unwound would be free if the k = n
    # term were missing. Compare a strategy that ends flat against one that does
    # not: the difference must be exactly the cost of unwinding the last position.
    S = torch.tensor([[100.0, 100.0, 120.0]])
    ends_flat = torch.tensor([[0.4, 0.0]])
    holds = torch.tensor([[0.4, 0.4]])
    c = 0.02
    cost_flat = float(transaction_cost(S, ends_flat, c))
    cost_hold = float(transaction_cost(S, holds, c))
    # ends_flat: +0.4 at 100, -0.4 at 100, 0 at 120  -> 0.02 * 80 = 1.6
    # holds:     +0.4 at 100,  0 at 100, -0.4 at 120 -> 0.02 * (40 + 48) = 1.76
    assert cost_flat == pytest.approx(1.6)
    assert cost_hold == pytest.approx(1.76)
    assert cost_hold > cost_flat


def test_costs_are_non_negative_and_increase_with_the_cost_rate():
    rng = np.random.default_rng(0)
    S = torch.as_tensor(100.0 * np.exp(np.cumsum(rng.standard_normal((200, 11)) * 0.02, axis=1)))
    deltas = torch.as_tensor(rng.uniform(-1.0, 1.0, (200, 10)))
    previous = torch.zeros(200, dtype=torch.float64)
    for c in [0.0, 0.001, 0.01, 0.05]:
        cost = transaction_cost(S, deltas, c)
        assert torch.all(cost >= 0.0)
        assert torch.all(cost >= previous - 1e-12)
        previous = cost
    # The cost is exactly linear in the rate.
    assert torch.allclose(
        transaction_cost(S, deltas, 0.02), 2.0 * transaction_cost(S, deltas, 0.01)
    )


def test_perfect_replication_of_a_linear_claim():
    # Hedging a forward and not a call: the payoff S_T - K is replicated
    # exactly by holding one share throughout, so with no costs the hedging gain
    # must equal S_T - S_0 and offset the payoff up to the constant S_0 - K.
    rng = np.random.default_rng(1)
    S = torch.as_tensor(100.0 * np.exp(np.cumsum(rng.standard_normal((500, 21)) * 0.03, axis=1)))
    S = S / S[:, :1] * 100.0
    deltas = torch.ones(500, 20, dtype=torch.float64)
    gain = hedging_gain(S, deltas)
    assert torch.allclose(gain, S[:, -1] - S[:, 0], atol=1e-9)


def test_hedging_gain_has_zero_mean_under_the_pricing_measure():
    # The discrete stochastic integral against a martingale is a martingale, so
    # its expectation is zero for any predictable strategy: including one that
    # depends on the path. This is the property the indifference price relies on.
    from deephedging.simulate import gbm_paths

    S = gbm_paths(200_000, 20, 1.0, 100.0, 0.25, seed=2)
    S_t = torch.as_tensor(S)
    # A deliberately path-dependent, non-trivial strategy.
    deltas = torch.tanh((S_t[:, :20] - 100.0) / 10.0)
    gain = hedging_gain(S_t, deltas).numpy()
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    assert abs(gain.mean()) < 4.0 * se, (gain.mean(), se)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="columns"):
        pnl(torch.zeros(5, 10), torch.zeros(5, 10), 100.0)
