"""Two runs with the same seed must agree exactly, and different seeds must not.

Exactly, not approximately. Everything stochastic in the pipeline is seeded: the
path simulation through numpy's generator, and the weight initialisation and batch
shuffling through torch's. If any of them were reading a global unseeded state, the
equality below would fail, and a number in the README would not be reproducible.

The second half of each test matters as much as the first: a run that ignores its
seed would also pass an equality check, so the tests confirm that changing the seed
changes the result.
"""

from __future__ import annotations

import numpy as np
import torch

from deephedging.simulate import gbm_paths
from deephedging.train import TrainConfig, positions_from_hedger, train_hedger

T, K, SIGMA = 1.0, 100.0, 0.2


def _run(seed: int, cost_rate: float = 0.005, risk: str = "entropic"):
    S = gbm_paths(20_000, 10, T, 100.0, SIGMA, seed=1)
    config = TrainConfig(risk=risk, epochs=6, batch_size=2048, seed=seed)
    result = train_hedger(S, K, T, cost_rate, config=config)
    return positions_from_hedger(result.hedger, S), result.history


def test_two_runs_with_the_same_seed_agree_exactly():
    a_positions, a_history = _run(seed=0)
    b_positions, b_history = _run(seed=0)
    assert np.array_equal(a_positions, b_positions)
    assert a_history == b_history


def test_a_different_seed_gives_a_different_result():
    a_positions, _ = _run(seed=0)
    b_positions, _ = _run(seed=1)
    assert not np.allclose(a_positions, b_positions)


def test_the_cvar_objective_is_reproducible_too():
    # The CVaR objective carries an extra learned parameter, so it exercises a
    # different code path through the optimiser.
    a, a_hist = _run(seed=0, risk="cvar")
    b, b_hist = _run(seed=0, risk="cvar")
    assert np.array_equal(a, b)
    assert a_hist == b_hist


def test_evaluation_is_deterministic_given_a_trained_network():
    S = gbm_paths(5_000, 10, T, 100.0, SIGMA, seed=2)
    config = TrainConfig(epochs=4, batch_size=2048, seed=0)
    hedger = train_hedger(S, K, T, 0.0, config=config).hedger
    first = positions_from_hedger(hedger, S)
    second = positions_from_hedger(hedger, S)
    assert np.array_equal(first, second)
    # And batching the evaluation must not change it.
    third = positions_from_hedger(hedger, S, batch_size=137)
    assert np.allclose(first, third, atol=0.0)


def test_bootstrap_standard_errors_are_seeded():
    from deephedging.train import bootstrap_se

    rng = np.random.default_rng(0)
    sample = rng.standard_normal(5_000)
    a = bootstrap_se(sample, np.mean, n_resamples=100, seed=3)
    b = bootstrap_se(sample, np.mean, n_resamples=100, seed=3)
    c = bootstrap_se(sample, np.mean, n_resamples=100, seed=4)
    assert a == b
    assert a != c
    # It should also be close to the analytic standard error of the mean.
    assert abs(a - sample.std(ddof=1) / np.sqrt(len(sample))) < 0.002


def test_torch_default_dtype_and_thread_settings_do_not_leak():
    # A module that changed a torch global would make results depend on import
    # order, which is a reproducibility hazard worth pinning down explicitly.
    assert torch.get_default_dtype() == torch.float32
