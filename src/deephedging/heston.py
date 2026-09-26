"""Semi-analytic Heston pricing, in the Albrecher et al. "little trap" form.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^2_t,   d<W^1, W^2>_t = rho dt

`xi` is the volatility of variance, renamed from the usual sigma so `sigma`
always means a Black-Scholes volatility here.

The characteristic function contains a complex logarithm. Heston's 1993 form
uses g1 with |g1| > 1, so as T grows the argument winds around the origin and a
principal-branch implementation jumps by 2 pi i, giving the integrand spurious
discontinuities and the quadrature a wrong price past a year or two. Albrecher,
Mayer, Schoutens and Tistaert (2007) rewrite it with g2 = 1/g1, |g2| < 1, which
keeps 1 - g2 exp(-d T) in the right half plane. Only g2 is implemented;
`tests/test_heston.py` checks prices stay smooth in T out to five years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters. Frozen so a calibration result cannot be mutated."""

    v0: float  # initial instantaneous variance
    kappa: float  # mean-reversion speed of the variance
    theta: float  # long-run variance
    xi: float  # volatility of variance
    rho: float  # correlation between the two Brownian motions

    def feller(self) -> float:
        """2 kappa theta - xi**2; non-negative means Feller holds.

        Under Feller the variance stays strictly positive; violated, it can
        touch zero, which the scheme in `simulate.py` has to cope with. Real
        calibrations violate it routinely, so this is reported, not enforced.
        """
        return 2.0 * self.kappa * self.theta - self.xi**2

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho], dtype=float)


def char_function(
    u: np.ndarray, T: float, r: float, q: float, p: HestonParams, S: float = 1.0
) -> np.ndarray:
    """E[exp(i u log S_T)] for the Heston model, little-trap form.

    `u` may be complex: P1 in `_p1_p2` is an expectation under the share
    measure and is obtained by evaluating this at u - i.
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0

    # d is a square root of a complex number; numpy's principal branch is fine
    # here, the branch problem is in the logarithm below, not in this root.
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi**2 * (1j * u + u**2))

    # g2 instead of g1: this single line is the "little trap".
    num = kappa - rho * xi * 1j * u - d
    den = kappa - rho * xi * 1j * u + d
    g = num / den

    exp_dT = np.exp(-d * T)
    # |g| < 1 and |exp_dT| <= 1, so 1 - g exp_dT never approaches the branch cut.
    C = (kappa * theta / xi**2) * (num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
    D = (v0 / xi**2) * num * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    drift = 1j * u * (np.log(S) + (r - q) * T)
    return np.exp(drift + C + D)


def _gauss_legendre_nodes(n: int, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on (0, upper].

    Fixed rather than adaptive quadrature, so a whole surface prices in one
    vectorised pass inside each calibration step. `tests/test_heston.py` pins
    the result against adaptive quadrature.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    # map [-1, 1] -> (0, upper]
    return 0.5 * upper * (x + 1.0), 0.5 * upper * w


def _p1_p2(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    n_nodes: int = 256,
    upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The two Heston probabilities in C = S e^{-qT} P1 - K e^{-rT} P2.

    P2 is the risk-neutral probability of finishing in the money, P1 the same
    event under the stock as numeraire. Both come from one characteristic
    function via f1(u) = phi(u - i) / phi(-i), with phi(-i) the forward.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if upper is None:
        # The integrand decays like exp(-u^2 v T / 2), so the cut-off frequency
        # scales as 1 / sqrt(v T) and a fixed limit is too low at short
        # maturity. At one month, a limit of 200 leaves a truncation error of
        # 2e-7, above the quadrature error and so the binding term.
        scale = np.sqrt(max(p.v0, p.theta, 1e-6) * max(T, 1e-8))
        upper = float(np.clip(20.0 / scale, 200.0, 5000.0))
    u, w = _gauss_legendre_nodes(n_nodes, upper)

    phi2 = char_function(u, T, r, q, p, S=S)  # shape (n_nodes,)
    forward = S * np.exp((r - q) * T)
    phi1 = char_function(u - 1j, T, r, q, p, S=S) / forward

    # Outer product over (strike, node). A few hundred strikes times a few
    # hundred nodes is a small matrix, so this stays comfortable in memory.
    phase = np.exp(-1j * np.outer(np.log(K), u))
    integrand1 = np.real(phase * phi1[None, :] / (1j * u)[None, :])
    integrand2 = np.real(phase * phi2[None, :] / (1j * u)[None, :])

    P1 = 0.5 + integrand1 @ w / np.pi
    P2 = 0.5 + integrand2 @ w / np.pi
    return P1, P2


def call_price(
    S: float,
    K: np.ndarray | float,
    T: float,
    r: float = 0.0,
    q: float = 0.0,
    p: HestonParams | None = None,
    n_nodes: int = 256,
) -> np.ndarray:
    """European call price under Heston, for one maturity and many strikes."""
    if p is None:
        raise ValueError("Heston parameters are required")
    if T <= 0.0:
        return np.maximum(np.asarray(K, dtype=float) * 0.0 + S - np.asarray(K, dtype=float), 0.0)
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    P1, P2 = _p1_p2(S, K_arr, T, r, q, p, n_nodes=n_nodes)
    price = S * np.exp(-q * T) * P1 - K_arr * np.exp(-r * T) * P2
    # Quadrature error can push a deep out-of-the-money price marginally
    # negative; clip to the no-arbitrage lower bound instead of pretending.
    price = np.maximum(price, np.maximum(S * np.exp(-q * T) - K_arr * np.exp(-r * T), 0.0))
    return price if np.ndim(K) else price[0]


def call_delta(
    S: float,
    K: np.ndarray | float,
    T: float,
    r: float = 0.0,
    q: float = 0.0,
    p: HestonParams | None = None,
    n_nodes: int = 256,
) -> np.ndarray:
    """dC/dS under Heston, exactly, with no finite differencing.

    The price is homogeneous of degree one in (S, K), so Euler gives
    C = S dC/dS + K dC/dK. Differentiating C = S e^{-qT} P1 - K e^{-rT} P2 in K
    leaves dC/dK = -e^{-rT} P2, and substituting,

        S dC/dS = C + K e^{-rT} P2 = S e^{-qT} P1  =>  dC/dS = e^{-qT} P1.

    So the delta is P1, which the pricer already computed.
    `tests/test_heston.py` checks it against a central difference.
    """
    if p is None:
        raise ValueError("Heston parameters are required")
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    if T <= 0.0:
        delta = (K_arr < S).astype(float)
    else:
        P1, _ = _p1_p2(S, K_arr, T, r, q, p, n_nodes=n_nodes)
        delta = np.exp(-q * T) * P1
    return delta if np.ndim(K) else delta[0]


def call_dv(
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    q: float = 0.0,
    p: HestonParams | None = None,
    h: float = 1e-5,
    n_nodes: int = 256,
) -> float:
    """dC/dv0, by central difference. Needed for the minimum-variance delta."""
    if p is None:
        raise ValueError("Heston parameters are required")
    up = HestonParams(p.v0 + h, p.kappa, p.theta, p.xi, p.rho)
    down = HestonParams(max(p.v0 - h, 1e-12), p.kappa, p.theta, p.xi, p.rho)
    c_up = float(call_price(S, K, T, r, q, up, n_nodes=n_nodes))
    c_down = float(call_price(S, K, T, r, q, down, n_nodes=n_nodes))
    return (c_up - c_down) / (up.v0 - down.v0)
