"""The hedging policy: a network mapping the observable state to a position.

Shapes:
    features   (n_paths, n_features)     state of every path at one date
    output     (n_paths,)                position each path should hold

The current position is a feature, not a convenience: with costs, moving from
0.4 to 0.6 and from 0.8 to 0.6 both cost something, so the optimal action
depends on where the portfolio already is and the value function needs
(t, S, v, delta_previous). Nothing after date k is ever fed in;
`tests/test_no_lookahead.py` checks that mechanically.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(n_inputs: int, n_hidden: int, n_layers: int) -> nn.Sequential:
    """Plain feedforward network with a scalar output, ReLU activations.

    No activation on the last layer: clipping the position to [0, 1] would hand
    the network the answer to the convergence test. The last layer is
    initialised to zero so the strategy starts flat rather than at a random
    position, whose first-batch losses dominate the entropic log-sum-exp.
    """
    layers: list[nn.Module] = []
    width = n_inputs
    for _ in range(n_layers):
        layers += [nn.Linear(width, n_hidden), nn.ReLU()]
        width = n_hidden
    last = nn.Linear(width, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class DeepHedger(nn.Module):
    """Maps simulated paths to a sequence of hedge positions.

    `per_date=False`: one network with time to maturity as a feature, so the
    parameter count does not grow with the number of dates.
    `per_date=True`: n independent networks, one per date, as in Buehler, Gonon,
    Teichmann and Wood (2019). The study reports both.
    """

    def __init__(
        self,
        n_steps: int,
        maturity: float,
        strike: float,
        use_variance: bool = False,
        n_hidden: int = 32,
        n_layers: int = 2,
        per_date: bool = False,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.maturity = maturity
        self.strike = strike
        self.use_variance = use_variance
        self.per_date = per_date

        # features: log-moneyness, position carried in, [instantaneous vol], [tau]
        n_features = 2 + int(use_variance) + int(not per_date)
        self.n_features = n_features

        if per_date:
            # ModuleList and not a plain list: otherwise net.parameters() misses
            # these weights and they are silently never trained.
            self.nets = nn.ModuleList([mlp(n_features, n_hidden, n_layers) for _ in range(n_steps)])
        else:
            self.net = mlp(n_features, n_hidden, n_layers)

    def features(self, S_k: torch.Tensor, v_k: torch.Tensor | None, prev: torch.Tensor, k: int):
        """State at date k. Inputs (n_paths,), output (n_paths, n_features)."""
        cols = [torch.log(S_k / self.strike), prev]
        if self.use_variance:
            if v_k is None:
                raise ValueError("this hedger expects a variance path")
            # Volatility, not variance: same order as the other features (0.2,
            # not 0.04), which keeps the first linear layer well conditioned.
            cols.append(torch.sqrt(v_k))
        if not self.per_date:
            # Fraction of the option's life remaining, so the feature is in [0, 1].
            tau = torch.full_like(S_k, (self.n_steps - k) / self.n_steps)
            cols.append(tau)
        return torch.stack(cols, dim=1)

    def forward(self, S: torch.Tensor, v: torch.Tensor | None = None) -> torch.Tensor:
        """Positions for every path and date. S is (n_paths, n_steps + 1),
        output is (n_paths, n_steps).

        The loop over k cannot be vectorised: `prev` is the output of date
        k - 1. Autograd records the unrolled graph, so the gradient flows back
        through all n decisions and the number of dates drives the training cost.
        """
        prev = torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
        positions = []
        for k in range(self.n_steps):
            v_k = None if v is None else v[:, k]
            x = self.features(S[:, k], v_k, prev, k)
            net = self.nets[k] if self.per_date else self.net
            prev = net(x).squeeze(-1)  # (n_paths, 1) -> (n_paths,)
            positions.append(prev)
        return torch.stack(positions, dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
