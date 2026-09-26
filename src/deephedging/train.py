"""Training the hedging policy, and evaluating any policy on held-out paths.

What is optimised: one number, rho(-PnL), over the network weights and (for
CVaR) the Rockafellar-Uryasev variable. No labels, no supervised target, and
the Black-Scholes delta is never shown to the network during training, which is
what makes `tests/test_convergence_delta.py` a real test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from deephedging.hedging import pnl, turnover
from deephedging.networks import DeepHedger
from deephedging.risk import evaluate_risk, make_objective


@dataclass
class TrainConfig:
    """Everything that defines a training run, so one object reproduces it."""

    risk: str = "entropic"  # "entropic" or "cvar"
    lam: float = 1.0  # entropic risk aversion
    alpha: float = 0.95  # CVaR level
    epochs: int = 40
    batch_size: int = 4096
    learning_rate: float = 3e-3
    cosine_decay: bool = True  # anneal the step size to zero over training
    seed: int = 0
    n_hidden: int = 32
    n_layers: int = 2
    per_date: bool = False
    verbose: bool = False


@dataclass
class TrainResult:
    hedger: DeepHedger
    history: list[float] = field(default_factory=list)
    config: TrainConfig = field(default_factory=TrainConfig)


def _to_tensor(a: np.ndarray | None) -> torch.Tensor | None:
    """float32 throughout the network.

    Half the memory and about twice the speed, with noise far below the Monte
    Carlo error at 10^5 paths. The exception is the entropic measure, where
    float32 overflows past 88, which is why `risk.entropic_risk` uses logsumexp.
    """
    if a is None:
        return None
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


def train_hedger(
    S: np.ndarray,
    strike: float,
    maturity: float,
    cost_rate: float = 0.0,
    v: np.ndarray | None = None,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Fit a hedging policy on simulated paths by minimising the risk of the loss."""
    config = config or TrainConfig()
    n_steps = S.shape[1] - 1

    # Weight init and batch shuffling draw from torch's global generator, so
    # seeding numpy alone would still leave two runs on the same data differing.
    torch.manual_seed(config.seed)

    hedger = DeepHedger(
        n_steps=n_steps,
        maturity=maturity,
        strike=strike,
        use_variance=v is not None,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        per_date=config.per_date,
    )
    objective, extra = make_objective(config.risk, lam=config.lam, alpha=config.alpha)
    optimizer = torch.optim.Adam(list(hedger.parameters()) + extra, lr=config.learning_rate)
    # The objective is a minibatch Monte Carlo estimate, so its gradient is
    # noisy even at the optimum and a constant step size leaves the parameters
    # rattling in a neighbourhood set by the step size. That rattle would look
    # like a failure to converge to the analytic delta; annealing removes it.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0.0)
        if config.cosine_decay
        else None
    )

    S_t = _to_tensor(S)
    v_t = _to_tensor(v)
    n_paths = S_t.shape[0]
    generator = torch.Generator().manual_seed(config.seed + 1)

    history: list[float] = []
    for epoch in range(config.epochs):
        # Reshuffling decorrelates successive minibatch gradients, which is
        # what SGD assumes.
        order = torch.randperm(n_paths, generator=generator)
        epoch_losses = []
        for start in range(0, n_paths, config.batch_size):
            idx = order[start : start + config.batch_size]
            S_b = S_t[idx]
            v_b = None if v_t is None else v_t[idx]

            optimizer.zero_grad()
            deltas = hedger(S_b, v_b)
            loss = objective(-pnl(S_b, deltas, strike, cost_rate))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        if scheduler is not None:
            scheduler.step()  # once per epoch, so T_max is in units of `epochs`

        history.append(float(np.mean(epoch_losses)))
        if config.verbose and (epoch % 5 == 0 or epoch == config.epochs - 1):
            print(f"  epoch {epoch + 1:3d}/{config.epochs}  objective {history[-1]:+.5f}")

    return TrainResult(hedger=hedger, history=history, config=config)


def positions_from_hedger(
    hedger: DeepHedger, S: np.ndarray, v: np.ndarray | None = None, batch_size: int = 20000
) -> np.ndarray:
    """Positions from a trained policy, shape (n_paths, n_steps).

    Batched: the unrolled graph over all paths at once is large even without
    gradients.
    """
    out = []
    with torch.no_grad():
        for start in range(0, S.shape[0], batch_size):
            S_b = _to_tensor(S[start : start + batch_size])
            v_b = None if v is None else _to_tensor(v[start : start + batch_size])
            out.append(hedger(S_b, v_b).numpy())
    return np.concatenate(out, axis=0)


def bootstrap_se(sample: np.ndarray, statistic, n_resamples: int = 400, seed: int = 0) -> float:
    """Standard error of a statistic, by the non-parametric bootstrap.

    A risk measure is a non-linear functional of the empirical distribution and
    has no analytic standard error. 400 resamples gives two significant figures.
    """
    rng = np.random.default_rng(seed)
    n = len(sample)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        values[b] = statistic(sample[rng.integers(0, n, n)])
    return float(np.std(values, ddof=1))


def evaluate_strategy(
    S: np.ndarray,
    deltas: np.ndarray,
    strike: float,
    cost_rate: float,
    risk: str = "entropic",
    lam: float = 1.0,
    alpha: float = 0.95,
    seed: int = 0,
    n_resamples: int = 400,
) -> dict:
    """Summarise a strategy on held-out paths, with Monte Carlo standard errors.

    At 10^5 paths a hedge's risk is known to about two decimals, so every number
    here carries a standard error.
    """
    S_t = _to_tensor(S)
    d_t = _to_tensor(deltas)
    with torch.no_grad():
        p = pnl(S_t, d_t, strike, cost_rate).numpy().astype(np.float64)
        turn = turnover(S_t, d_t).numpy().astype(np.float64)
        gross = pnl(S_t, d_t, strike, 0.0).numpy().astype(np.float64)

    loss = -p
    n = len(loss)

    def risk_of(sample: np.ndarray) -> float:
        return evaluate_risk(torch.as_tensor(sample), risk, lam=lam, alpha=alpha)

    return {
        "risk": risk_of(loss),
        "risk_se": bootstrap_se(loss, risk_of, n_resamples=n_resamples, seed=seed),
        "mean_pnl": float(p.mean()),
        "mean_pnl_se": float(p.std(ddof=1) / np.sqrt(n)),
        "std_pnl": float(p.std(ddof=1)),
        "cvar95": risk_of(loss) if risk == "cvar" else float("nan"),
        "mean_turnover": float(turn.mean()),
        "mean_turnover_se": float(turn.std(ddof=1) / np.sqrt(n)),
        "total_cost": float((gross - p).mean()),
        "n_paths": n,
    }


def paired_risk_difference(
    S: np.ndarray,
    deltas_a: np.ndarray,
    deltas_b: np.ndarray,
    strike: float,
    cost_rate: float,
    risk: str = "entropic",
    lam: float = 1.0,
    alpha: float = 0.95,
    seed: int = 0,
    n_resamples: int = 400,
) -> dict:
    """rho(A) - rho(B) with a standard error from a paired bootstrap.

    Both strategies run on the same paths, so resampling jointly cancels the
    shared Monte Carlo noise and gives a much smaller standard error than
    combining two independent ones. This is the number that decides whether the
    learned strategy beats the delta hedge.
    """
    S_t = _to_tensor(S)
    with torch.no_grad():
        loss_a = -pnl(S_t, _to_tensor(deltas_a), strike, cost_rate).numpy().astype(np.float64)
        loss_b = -pnl(S_t, _to_tensor(deltas_b), strike, cost_rate).numpy().astype(np.float64)

    def risk_of(sample: np.ndarray) -> float:
        return evaluate_risk(torch.as_tensor(sample), risk, lam=lam, alpha=alpha)

    point = risk_of(loss_a) - risk_of(loss_b)
    rng = np.random.default_rng(seed)
    n = len(loss_a)
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)  # the same indices for both strategies
        diffs[b] = risk_of(loss_a[idx]) - risk_of(loss_b[idx])
    return {"difference": point, "se": float(np.std(diffs, ddof=1))}
