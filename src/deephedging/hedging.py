"""Terminal PnL of a short European call, hedged, with proportional costs.

    PnL = -payoff(S_T)
          + sum_{k=0}^{n-1} delta_k (S_{k+1} - S_k)
          - c sum_{k=0}^{n} |delta_k - delta_{k-1}| S_k

with delta_{-1} = 0 and delta_n = 0, so the terminal liquidation is charged.
The option premium is not included: rho(-PnL) is then a risk measure of the
hedging shortfall alone, and the price comes out of `pricing.py` separately.
Everything is in torch because the training loop differentiates through it.
"""

from __future__ import annotations

import torch


def trade_sizes(deltas: torch.Tensor) -> torch.Tensor:
    """The n + 1 trades implied by n held positions.

    `deltas` is (n_paths, n_steps), entry k the position held over
    [t_k, t_{k+1}). Returns (n_paths, n_steps + 1) with the flat start and the
    terminal liquidation included.
    """
    zeros = torch.zeros_like(deltas[:, :1])
    padded = torch.cat([zeros, deltas, zeros], dim=1)  # delta_{-1}, ..., delta_n
    return padded[:, 1:] - padded[:, :-1]


def transaction_cost(S: torch.Tensor, deltas: torch.Tensor, cost_rate: float) -> torch.Tensor:
    """c * sum |trade| * S, the standard proportional (half-spread) cost model."""
    if cost_rate == 0.0:
        return torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
    return cost_rate * (trade_sizes(deltas).abs() * S).sum(dim=1)


def hedging_gain(S: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """sum_k delta_k (S_{k+1} - S_k).

    Has expectation zero under the pricing measure, which the
    indifference-price test relies on.
    """
    increments = S[:, 1:] - S[:, :-1]
    return (deltas * increments).sum(dim=1)


def pnl(
    S: torch.Tensor,
    deltas: torch.Tensor,
    strike: float,
    cost_rate: float = 0.0,
) -> torch.Tensor:
    """Terminal PnL per path, shape (n_paths,).

    S is (n_paths, n_steps + 1), deltas is (n_paths, n_steps).
    """
    if S.shape[1] != deltas.shape[1] + 1:
        raise ValueError(
            f"expected S with {deltas.shape[1] + 1} columns to match "
            f"{deltas.shape[1]} positions, got {S.shape[1]}"
        )
    payoff = torch.clamp(S[:, -1] - strike, min=0.0)
    return -payoff + hedging_gain(S, deltas) - transaction_cost(S, deltas, cost_rate)


def turnover(S: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Traded notional per path, sum_k |delta_k - delta_{k-1}| S_k."""
    return (trade_sizes(deltas).abs() * S).sum(dim=1)
