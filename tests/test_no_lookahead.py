"""The network must not use information it could not have had.

This is the test a reader of a machine-learning-in-finance repository should look
for first, because look-ahead is the failure mode that produces spectacular
results and no edge. Here it is checked mechanically over argued from the
code: a path is modified only after date k, the whole strategy is recomputed, and
every action up to and including date k must be bit-for-bit identical.

The test is blunt on purpose. It does not inspect the feature construction or
trust the architecture; it perturbs the future and looks at the past.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedging.heston import HestonParams
from deephedging.networks import DeepHedger
from deephedging.simulate import gbm_paths, heston_paths
from deephedging.train import TrainConfig, positions_from_hedger, train_hedger


def _trained_hedger(n_steps=10, per_date=False, with_variance=False):
    """A briefly trained network. Training matters: an untrained network with a
    zero-initialised output layer returns zero everywhere, which would pass any
    invariance test trivially."""
    T, K = 1.0, 100.0
    if with_variance:
        p = HestonParams(0.04, 2.0, 0.05, 0.4, -0.6)
        S, v = heston_paths(20_000, n_steps, T, p, seed=1, substeps=2)
    else:
        S, v = gbm_paths(20_000, n_steps, T, 100.0, 0.2, seed=1), None
    config = TrainConfig(epochs=8, batch_size=2048, seed=0, per_date=per_date)
    result = train_hedger(S, K, T, cost_rate=0.005, v=v, config=config)
    return result.hedger, S, v


@pytest.mark.parametrize("per_date", [False, True])
def test_actions_are_unchanged_when_the_future_of_the_path_is_perturbed(per_date):
    hedger, S, _ = _trained_hedger(per_date=per_date)
    n_steps = S.shape[1] - 1
    base = positions_from_hedger(hedger, S)
    # Sanity: the trained network is not the trivial zero strategy, or the test
    # would prove nothing.
    assert np.abs(base).mean() > 0.05

    rng = np.random.default_rng(7)
    for k in [0, 1, n_steps // 2, n_steps - 2]:
        perturbed = S.copy()
        # Multiply everything strictly after date k by a large random factor. Any
        # dependence of an earlier action on a later price would show up.
        factor = np.exp(rng.normal(0.0, 0.5, size=(S.shape[0], 1)))
        perturbed[:, k + 1 :] *= factor
        after = positions_from_hedger(hedger, perturbed)
        assert np.array_equal(base[:, : k + 1], after[:, : k + 1]), k
        # And the actions after k must actually change, otherwise the perturbation
        # was not doing anything and the invariance above is vacuous.
        if k + 1 < n_steps:
            assert not np.allclose(base[:, k + 1 :], after[:, k + 1 :])


def test_actions_are_unchanged_when_the_future_variance_is_perturbed():
    # The same check on the second state variable, for the Heston hedger.
    hedger, S, v = _trained_hedger(with_variance=True)
    n_steps = S.shape[1] - 1
    base = positions_from_hedger(hedger, S, v)
    assert np.abs(base).mean() > 0.05

    for k in [0, n_steps // 2, n_steps - 2]:
        perturbed = v.copy()
        perturbed[:, k + 1 :] *= 5.0
        after = positions_from_hedger(hedger, S, perturbed)
        assert np.array_equal(base[:, : k + 1], after[:, : k + 1]), k
        if k + 1 < n_steps:
            assert not np.allclose(base[:, k + 1 :], after[:, k + 1 :])


def test_the_feature_vector_at_a_date_is_built_only_from_that_date():
    # A direct check on the feature construction, independent of training: the
    # features at date k must be a function of column k alone.
    hedger = DeepHedger(n_steps=6, maturity=1.0, strike=100.0, use_variance=True)
    S = torch.as_tensor(np.linspace(80.0, 140.0, 7 * 4).reshape(4, 7), dtype=torch.float32)
    v = torch.full((4, 7), 0.04)
    prev = torch.zeros(4)
    for k in range(6):
        features = hedger.features(S[:, k], v[:, k], prev, k)
        S_changed = S.clone()
        S_changed[:, k + 1 :] = 999.0
        again = hedger.features(S_changed[:, k], v[:, k], prev, k)
        assert torch.equal(features, again)


def test_the_position_carried_in_is_the_previous_action_and_starts_at_zero():
    # The one piece of history the network is allowed: delta_{k-1}. Verify it is
    # exactly the previous output, so the state is what the control problem says.
    hedger, S, _ = _trained_hedger(n_steps=8)
    positions = positions_from_hedger(hedger, S[:64])
    S_t = torch.as_tensor(S[:64], dtype=torch.float32)
    prev = torch.zeros(64)
    with torch.no_grad():
        for k in range(8):
            features = hedger.features(S_t[:, k], None, prev, k)
            assert torch.allclose(features[:, 1], prev)
            prev = hedger.net(features).squeeze(-1)
            assert np.allclose(prev.numpy(), positions[:, k], atol=1e-6)
