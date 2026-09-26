"""Figures for the README.

Static PNGs, one message per figure, no second y-axis anywhere: two scales in
one chart ask the reader to compare arbitrary units, so where two quantities
need to be seen together they get two panels.

The four series colours are a fixed categorical order (delta hedge, deep hedge,
minimum-variance delta, market), assigned by identity and never recycled, so the
same object is the same colour everywhere. Checked for colour-vision-deficiency
separation and for 3:1 contrast against the chart surface, and every series is
labelled as well, so identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a CI runner
import matplotlib.pyplot as plt
import numpy as np

DELTA = "#2a78d6"  # blue: the analytic delta hedge
DEEP = "#eb6834"  # orange: the learned strategy
MINVAR = "#199e70"  # green: the minimum-variance delta
MARKET = "#4a3aa7"  # violet: market data
GRID = "#d8d8d4"
INK = "#1a1a19"
MUTED = "#6b6b66"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "font.size": 10,
    }
)


def _finish(fig, path: Path, top: float = 1.0) -> Path:
    """Save and close. `top` reserves room for a suptitle, which tight_layout
    otherwise clips off the canvas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(path)
    plt.close(fig)
    return path


def learned_vs_analytic_delta(
    learned: np.ndarray,
    analytic: np.ndarray,
    log_moneyness: np.ndarray,
    time_to_maturity: np.ndarray,
    path: Path,
    title: str = "",
) -> Path:
    """The headline figure: does the network recover the Black-Scholes delta?

    Left panel, the comparison as a density rather than a scatter: a million
    points plotted individually would be a solid block and hide whether the mass
    sits on the diagonal. Right panel, both strategies against log-moneyness at
    three times to maturity, where a systematic difference would show.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    density = left.hexbin(
        analytic.ravel(), learned.ravel(), gridsize=70, bins="log", cmap="Blues", mincnt=1
    )
    bar = fig.colorbar(density, ax=left, pad=0.02)
    bar.set_label("states visited (log scale)", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    left.plot([0, 1], [0, 1], color=INK, linewidth=1.2, linestyle="--", label="exact agreement")
    left.set_xlabel("Black-Scholes delta")
    left.set_ylabel("learned position")
    left.set_xlim(-0.02, 1.02)
    left.set_ylim(-0.02, 1.02)
    left.legend(loc="upper left")
    left.set_title("Learned position against the analytic delta")

    # Two slices in time to maturity, each a thin band so the curve is a genuine
    # conditional mean instead of a mixture over the whole life of the option. Two
    # rather than three: colour already carries the strategy and dash the maturity,
    # and a six-entry legend covered the curves it was labelling.
    # Interior dates only. At the very first date every path sits at the initial
    # spot, so the conditional curve there is a single point and plots as nothing;
    # at the very last it is a step. Two interior slices show the shape.
    taus = np.unique(time_to_maturity)
    chosen = [taus[2 * len(taus) // 3], taus[max(len(taus) // 4, 1)]]
    styles = ["-", "--"]
    for tau, style in zip(chosen, styles, strict=False):
        mask = np.isclose(time_to_maturity, tau)
        x = log_moneyness[mask]
        order = np.argsort(x)
        x = x[order]
        # Bin the two strategies on the same grid and plot the conditional means.
        edges = np.quantile(x, np.linspace(0.005, 0.995, 45))
        idx = np.clip(np.searchsorted(edges, x), 0, len(edges) - 1)
        centres, deep_mean, delta_mean = [], [], []
        for b in range(len(edges)):
            sel = idx == b
            if sel.sum() < 40:
                continue
            centres.append(x[sel].mean())
            deep_mean.append(learned[mask][order][sel].mean())
            delta_mean.append(analytic[mask][order][sel].mean())
        label_tau = f"{tau:.2f}y to maturity"
        right.plot(centres, delta_mean, style, color=DELTA, label=f"delta, {label_tau}")
        right.plot(centres, deep_mean, style, color=DEEP, label=f"deep, {label_tau}")

    right.axhline(0.0, color=GRID, linewidth=0.6)
    right.set_xlabel("log-moneyness  log(S / K)")
    right.set_ylabel("position in the underlying")
    right.grid(True, axis="y")
    right.legend(ncol=1, loc="upper left", fontsize=8)
    right.set_title("By moneyness, at two times to maturity")

    if title:
        fig.suptitle(title, fontsize=12)
    return _finish(fig, path, top=0.90 if title else 1.0)


def pnl_distributions(
    pnl_delta: np.ndarray,
    pnl_deep: np.ndarray,
    path: Path,
    cost_rate: float,
    risk_label: str = "",
) -> Path:
    """Profit and loss of the two strategies on the same paths, under costs.

    Overlaid histograms on a single axis: both are in units of the underlying
    and share a scale. The lower panel shows the left tail alone, which is what a
    convex risk measure looks at and is invisible at a scale set by the mode.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 6.2), height_ratios=[2, 1])

    lo = float(min(np.quantile(pnl_delta, 0.001), np.quantile(pnl_deep, 0.001)))
    hi = float(max(np.quantile(pnl_delta, 0.999), np.quantile(pnl_deep, 0.999)))
    bins = np.linspace(lo, hi, 140)
    for data, colour, label in [
        (pnl_delta, DELTA, "delta hedge"),
        (pnl_deep, DEEP, "deep hedge"),
    ]:
        top.hist(data, bins=bins, histtype="step", color=colour, linewidth=1.8, label=label)
        top.axvline(np.mean(data), color=colour, linewidth=1.0, linestyle=":")
    top.set_xlabel("terminal profit and loss of the hedge, in units of the underlying")
    top.set_ylabel("paths")
    top.grid(True, axis="y")
    top.legend()
    top.set_title(
        f"Hedging profit and loss at a cost rate of {cost_rate * 1e4:.0f} basis points"
        + (f", {risk_label}" if risk_label else "")
    )

    tail_hi = float(np.quantile(np.concatenate([pnl_delta, pnl_deep]), 0.05))
    tail_bins = np.linspace(lo, tail_hi, 60)
    for data, colour, label in [
        (pnl_delta, DELTA, "delta hedge"),
        (pnl_deep, DEEP, "deep hedge"),
    ]:
        bottom.hist(data, bins=tail_bins, histtype="step", color=colour, linewidth=1.8, label=label)
    bottom.set_xlabel("the worst 5% of outcomes")
    bottom.set_ylabel("paths")
    bottom.grid(True, axis="y")
    bottom.legend()
    return _finish(fig, path)


def risk_and_turnover_against_cost(
    cost_rates: np.ndarray,
    rows: dict[str, dict[str, np.ndarray]],
    path: Path,
    risk_label: str,
) -> Path:
    """Risk and turnover against the cost rate, as two panels and never two axes.

    `rows` maps a strategy name to dicts of arrays: risk, risk_se, turnover,
    turnover_se. Error bars are the bootstrap standard errors; the claim of the
    study is a difference between two of these curves.
    """
    colours = {"delta hedge": DELTA, "deep hedge": DEEP, "minimum-variance delta": MINVAR}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bp = cost_rates * 1e4

    for name, series in rows.items():
        colour = colours.get(name, MUTED)
        left.errorbar(
            bp,
            series["risk"],
            yerr=series["risk_se"],
            color=colour,
            marker="o",
            markersize=5,
            capsize=3,
            label=name,
        )
        right.errorbar(
            bp,
            series["turnover"],
            yerr=series["turnover_se"],
            color=colour,
            marker="o",
            markersize=5,
            capsize=3,
            label=name,
        )

    left.set_xlabel("proportional cost rate (basis points)")
    left.set_ylabel(f"{risk_label} of the hedging loss")
    left.grid(True, axis="y")
    left.legend()
    left.set_title("Risk of the hedging loss")

    right.set_xlabel("proportional cost rate (basis points)")
    right.set_ylabel("traded notional per path")
    right.grid(True, axis="y")
    right.legend()
    right.set_title("How much each strategy trades")
    return _finish(fig, path)


def risk_advantage(
    cost_rates: np.ndarray,
    difference: np.ndarray,
    standard_error: np.ndarray,
    path: Path,
    risk_label: str,
) -> Path:
    """The headline result: how much lower the learned strategy's risk is.

    The paired difference with its own standard error, rather than two curves to
    subtract by eye, and with zero marked: whether the difference is
    distinguishable from zero is the finding.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bp = cost_rates * 1e4
    ax.axhline(0.0, color=INK, linewidth=1.0, linestyle="--", label="no improvement")
    ax.errorbar(
        bp,
        difference,
        yerr=1.96 * np.asarray(standard_error),
        color=DEEP,
        marker="o",
        markersize=5,
        capsize=3,
        label="deep hedge minus delta hedge",
    )
    ax.fill_between(
        bp,
        np.asarray(difference) - 1.96 * np.asarray(standard_error),
        np.asarray(difference) + 1.96 * np.asarray(standard_error),
        color=DEEP,
        alpha=0.12,
    )
    ax.set_xlabel("proportional cost rate (basis points)")
    ax.set_ylabel(f"change in {risk_label} (negative is better)")
    ax.grid(True, axis="y")
    ax.legend()
    ax.set_title("Advantage of the learned strategy, with 95% intervals")
    return _finish(fig, path)


def discretisation_rate(ns: np.ndarray, stds: np.ndarray, slope: float, path: Path) -> Path:
    """Hedging error of the delta strategy against the number of rebalancing dates.

    Log-log, with the fitted slope in the legend against the theoretical
    n^{-1/2} line. The point is the exponent, not that the curve goes down.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ns = np.asarray(ns, dtype=float)
    ax.loglog(ns, stds, "o-", color=DELTA, markersize=6, label=f"measured, slope {slope:+.3f}")
    reference = stds[0] * (ns / ns[0]) ** (-0.5)
    ax.loglog(ns, reference, "--", color=INK, linewidth=1.2, label="theory, slope $-1/2$")
    ax.set_xlabel("rebalancing dates $n$")
    ax.set_ylabel("standard deviation of the replication error")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("Discretisation error of the delta hedge, frictionless")
    return _finish(fig, path)


def calibration_fit(fit: object, path: Path, rmse_iv: float, source: str) -> Path:
    """Market and model implied volatility against strike, one panel per maturity.

    Market as points, model as a line, so the residual is visible without a
    second plot. Panels share a y-axis, so the term structure of the level shows
    alongside the fit at each maturity.
    """
    import pandas as pd

    assert isinstance(fit, pd.DataFrame)
    maturities = sorted(fit["maturity"].unique())
    n = len(maturities)
    # Balance the grid instead of filling rows of four and leaving three blanks:
    # for n panels the squarest layout is ceil(sqrt(n)) columns.
    cols = min(4, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols, 2.8 * rows), sharey=True, squeeze=False
    )
    for ax, T in zip(axes.ravel(), maturities, strict=False):
        slice_ = fit[fit["maturity"] == T].sort_values("strike")
        x = slice_["strike"] / slice_["forward"]
        ax.plot(x, slice_["market_iv"] * 100, "o", color=MARKET, markersize=4, label="market")
        ax.plot(x, slice_["model_iv"] * 100, "-", color=DEEP, label="Heston fit")
        ax.set_title(f"{T * 365:.0f} days", fontsize=10)
        ax.grid(True, axis="y")
        ax.set_xlabel("strike / forward")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    # sharey hides the tick labels but not the axis label, so label the first panel
    # of every row and not only the very first.
    for r in range(rows):
        axes[r, 0].set_ylabel("implied volatility (%)")
    axes[0, 0].legend()
    fig.suptitle(f"Heston calibrated to {source}: RMSE {rmse_iv * 100:.2f} vol points", fontsize=12)
    return _finish(fig, path, top=0.92)


def indifference_price_convergence(
    lambdas: np.ndarray,
    prices: np.ndarray,
    intercept: float,
    slope: float,
    bs_price: float,
    path: Path,
) -> Path:
    """Entropic indifference price against risk aversion, with the fitted intercept.

    The statement is about the limit at lambda = 0, which cannot be computed
    directly, so the figure shows the linear fit whose intercept estimates it and
    the Black-Scholes price it must equal.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid = np.linspace(0.0, float(np.max(lambdas)) * 1.05, 50)
    ax.plot(lambdas, prices, "o", color=DEEP, markersize=6, label="indifference price")
    ax.plot(grid, intercept + slope * grid, "-", color=DEEP, linewidth=1.4, label="linear fit")
    ax.axhline(
        bs_price,
        color=DELTA,
        linewidth=1.6,
        linestyle="--",
        label=f"Black-Scholes price {bs_price:.4f}",
    )
    ax.plot([0.0], [intercept], "*", color=INK, markersize=12, label=f"intercept {intercept:.4f}")
    ax.set_xlabel(r"risk aversion $\lambda$")
    ax.set_ylabel("price")
    ax.grid(True, axis="y")
    ax.legend()
    ax.set_title(r"Entropic indifference price as $\lambda \to 0$, zero costs")
    return _finish(fig, path)


def delta_recovery_rate(ns: np.ndarray, deviations: np.ndarray, slope: float, path: Path) -> Path:
    """Mean absolute deviation from the analytic delta against rebalancing dates.

    As a rate rather than a single number: perfect replication only exists in
    the limit, so what is checkable at finite n is that the gap closes, and how
    fast.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ns = np.asarray(ns, dtype=float)
    ax.loglog(ns, deviations, "o-", color=DEEP, markersize=6, label=f"measured, slope {slope:+.3f}")
    ax.set_xlabel("rebalancing dates $n$")
    ax.set_ylabel("mean $|$learned $-$ Black-Scholes delta$|$")
    ax.grid(True, which="both")
    ax.legend()
    ax.set_title("Convergence of the learned strategy to the analytic delta")
    return _finish(fig, path)
