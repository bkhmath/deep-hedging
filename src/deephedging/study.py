"""The experiments, each one stating its hypothesis before it produces a number.

Every function returns a plain dictionary and writes at most one figure;
`scripts/run_study.py` assembles the report, and numbers land in it whether they
are favourable or not.

In the order they should be read:

    1. `discretisation_rate_experiment`: the delta hedge alone, frictionless.
       The replication error should decay like n^{-1/2}.
    2. `delta_recovery_experiment`: refining the grid, the learned position
       should converge to the analytic delta.
    3. `architecture_experiment`: one shared network with time as an input
       against one network per date.
    4. `indifference_price_experiment`: the entropic indifference price should
       tend to the Black-Scholes price as risk aversion goes to zero.
    5. `cost_sweep_experiment`: the result. With costs the learned strategy
       should reach a lower risk than the delta hedge, by trading less.
    6. `heston_experiment`: the same under stochastic volatility at calibrated
       parameters, against the model delta and the minimum-variance delta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deephedging import black_scholes as bs
from deephedging import plots
from deephedging.benchmarks import bs_delta_positions, heston_delta_positions
from deephedging.hedging import pnl
from deephedging.heston import HestonParams
from deephedging.pricing import fit_lambda_intercept, indifference_price
from deephedging.simulate import gbm_paths, heston_paths
from deephedging.train import (
    TrainConfig,
    _to_tensor,
    evaluate_strategy,
    paired_risk_difference,
    positions_from_hedger,
    train_hedger,
)

# One set of market conventions for the Black-Scholes experiments, fixed here so
# that every experiment is quoted on the same option.
MATURITY = 1.0
STRIKE = 100.0
SPOT = 100.0
SIGMA = 0.2


def _delta_hedge_error_std(n_steps: int, n_paths: int, seed: int) -> tuple[float, float, float]:
    """Standard deviation, mean and its standard error of the replication error."""
    import torch

    S = gbm_paths(n_paths, n_steps, MATURITY, SPOT, SIGMA, seed=seed)
    deltas = bs_delta_positions(S, STRIKE, MATURITY, SIGMA)
    with torch.no_grad():
        terminal = pnl(_to_tensor(S), _to_tensor(deltas), STRIKE, 0.0).double().numpy()
    # The premium received at inception is the Black-Scholes price, so the
    # replication error is the terminal wealth plus that premium.
    error = terminal + float(bs.call_price(SPOT, STRIKE, MATURITY, 0.0, 0.0, SIGMA))
    return (
        float(error.std(ddof=1)),
        float(error.mean()),
        float(error.std(ddof=1) / np.sqrt(len(error))),
    )


def discretisation_rate_experiment(
    figure_dir: Path, n_paths: int = 100_000, ns: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
) -> dict:
    """Hypothesis: the delta hedge's replication error decays like n^{-1/2}.

    This is a theorem about the Euler discretisation of a stochastic integral, and it
    is the cleanest check that the simulator, the wealth equation and the analytic
    delta are mutually consistent: a wrong sign or an off-by-one in the time grid
    leaves the error decreasing but at the wrong rate, so the exponent is fitted.
    """
    stds, means, ses = [], [], []
    for n in ns:
        std, mean, se = _delta_hedge_error_std(n, n_paths, seed=1000 + n)
        stds.append(std)
        means.append(mean)
        ses.append(se)
    slope = float(np.polyfit(np.log(ns), np.log(stds), 1)[0])
    figure = plots.discretisation_rate(
        np.array(ns), np.array(stds), slope, figure_dir / "discretisation_rate.png"
    )
    return {
        "ns": list(ns),
        "stds": stds,
        "means": means,
        "mean_ses": ses,
        "slope": slope,
        "figure": figure,
    }


def delta_recovery_experiment(
    figure_dir: Path,
    ns: tuple[int, ...] = (15, 30, 60),
    n_paths: int = 200_000,
    n_test: int = 100_000,
    epochs: int = 60,
    lam: float = 1.0,
    seed: int = 0,
) -> dict:
    """Hypothesis: the learned position converges to the Black-Scholes delta.

    Without costs and on a fine enough grid the hedging error goes to zero, and
    the strategy attaining it is unique, so the minimiser of any risk measure is
    the delta hedge. The network never sees a delta in training.

    The risk aversion cannot be taken to zero to make the test easier: as
    lambda -> 0 the entropic measure tends to E[loss], the expected hedging gain
    is zero for every predictable strategy, and the delta stops being
    identified. So lambda is fixed and the grid is refined instead.
    """
    rows = []
    for n in ns:
        S_train = gbm_paths(n_paths, n, MATURITY, SPOT, SIGMA, seed=1)
        S_test = gbm_paths(n_test, n, MATURITY, SPOT, SIGMA, seed=2)
        analytic = bs_delta_positions(S_test, STRIKE, MATURITY, SIGMA)

        config = TrainConfig(risk="entropic", lam=lam, epochs=epochs, seed=seed)
        result = train_hedger(S_train, STRIKE, MATURITY, 0.0, config=config)
        learned = positions_from_hedger(result.hedger, S_test)

        deviation = np.abs(learned - analytic)
        deep = evaluate_strategy(S_test, learned, STRIKE, 0.0, "entropic", lam)
        delta = evaluate_strategy(S_test, analytic, STRIKE, 0.0, "entropic", lam)
        rows.append(
            {
                "n": n,
                "mean_abs_dev": float(deviation.mean()),
                "median_abs_dev": float(np.median(deviation)),
                "p95_abs_dev": float(np.quantile(deviation, 0.95)),
                "p99_abs_dev": float(np.quantile(deviation, 0.99)),
                "max_abs_dev": float(deviation.max()),
                "deep_risk": deep["risk"],
                "deep_risk_se": deep["risk_se"],
                "delta_risk": delta["risk"],
                "delta_risk_se": delta["risk_se"],
                "n_parameters": result.hedger.n_parameters(),
            }
        )
        if n == ns[-1]:
            # Figures only for the finest grid, which is where the claim is made.
            dt = MATURITY / n
            tau = MATURITY - dt * np.arange(n)
            log_moneyness = np.log(S_test[:, :n] / STRIKE)
            keep = slice(0, min(20_000, n_test))  # a subsample keeps the PNG small
            plots.learned_vs_analytic_delta(
                learned[keep],
                analytic[keep],
                log_moneyness[keep],
                np.repeat(tau[None, :], log_moneyness[keep].shape[0], axis=0),
                figure_dir / "learned_vs_delta.png",
                title=f"Black-Scholes, zero costs, n = {n} rebalancing dates",
            )

    devs = np.array([r["mean_abs_dev"] for r in rows])
    slope = float(np.polyfit(np.log(np.array(ns, dtype=float)), np.log(devs), 1)[0])
    figure = plots.delta_recovery_rate(
        np.array(ns), devs, slope, figure_dir / "delta_recovery_rate.png"
    )
    return {"rows": rows, "slope": slope, "figure": figure, "lam": lam}


def architecture_experiment(
    n_steps: int = 30,
    n_paths: int = 200_000,
    n_test: int = 100_000,
    epochs: int = 60,
    cost_rate: float = 0.002,
    lam: float = 1.0,
    seed: int = 0,
) -> dict:
    """Hypothesis: one shared network with time as an input matches n per-date networks.

    Buehler et al. use a separate network at every date. The shared-weights
    alternative says the optimal strategy is one function of (time, state), as
    the continuous-time problem asserts, so it should lose nothing at a fraction
    of the parameters. Run under costs.
    """
    S_train = gbm_paths(n_paths, n_steps, MATURITY, SPOT, SIGMA, seed=1)
    S_test = gbm_paths(n_test, n_steps, MATURITY, SPOT, SIGMA, seed=2)
    analytic = bs_delta_positions(S_test, STRIKE, MATURITY, SIGMA)

    out = {}
    positions = {}
    for label, per_date in [("shared weights", False), ("one network per date", True)]:
        config = TrainConfig(risk="entropic", lam=lam, epochs=epochs, seed=seed, per_date=per_date)
        result = train_hedger(S_train, STRIKE, MATURITY, cost_rate, config=config)
        learned = positions_from_hedger(result.hedger, S_test)
        positions[label] = learned
        summary = evaluate_strategy(S_test, learned, STRIKE, cost_rate, "entropic", lam)
        summary["n_parameters"] = result.hedger.n_parameters()
        out[label] = summary

    out["delta hedge"] = evaluate_strategy(S_test, analytic, STRIKE, cost_rate, "entropic", lam)
    out["difference"] = paired_risk_difference(
        S_test,
        positions["one network per date"],
        positions["shared weights"],
        STRIKE,
        cost_rate,
        "entropic",
        lam,
    )
    out["cost_rate"] = cost_rate
    out["n_steps"] = n_steps
    return out


def indifference_price_experiment(
    figure_dir: Path,
    n_steps: int = 30,
    n_paths: int = 200_000,
    n_test: int = 200_000,
    epochs: int = 40,
    lambdas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4),
    seed: int = 0,
) -> dict:
    """Hypothesis: the entropic indifference price tends to the Black-Scholes price.

    rho_lambda(X) = E[X] + (lambda/2) Var(X) + O(lambda^2), and the expected
    hedging gain is zero under the pricing measure, so at lambda = 0 the price is
    the risk-neutral expectation of the payoff, the Black-Scholes price, whatever
    the hedge does. The slope measures the unhedgeable residual variance, so one
    linear fit tests the limit and reads off the residual as a by-product.
    """
    S_train = gbm_paths(n_paths, n_steps, MATURITY, SPOT, SIGMA, seed=1)
    S_test = gbm_paths(n_test, n_steps, MATURITY, SPOT, SIGMA, seed=2)
    reference = float(bs.call_price(SPOT, STRIKE, MATURITY, 0.0, 0.0, SIGMA))

    prices, ses = [], []
    for lam in lambdas:
        config = TrainConfig(risk="entropic", lam=lam, epochs=epochs, seed=seed)
        result = train_hedger(S_train, STRIKE, MATURITY, 0.0, config=config)
        learned = positions_from_hedger(result.hedger, S_test)
        prices.append(indifference_price(S_test, learned, STRIKE, 0.0, "entropic", lam))
        ses.append(evaluate_strategy(S_test, learned, STRIKE, 0.0, "entropic", lam)["risk_se"])

    intercept, slope = fit_lambda_intercept(np.array(lambdas), np.array(prices))
    figure = plots.indifference_price_convergence(
        np.array(lambdas),
        np.array(prices),
        intercept,
        slope,
        reference,
        figure_dir / "indifference_price.png",
    )
    return {
        "lambdas": list(lambdas),
        "prices": prices,
        "price_ses": ses,
        "intercept": intercept,
        "slope": slope,
        "bs_price": reference,
        "figure": figure,
    }


def cost_sweep_experiment(
    figure_dir: Path,
    cost_rates: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01),
    risks: tuple[str, ...] = ("entropic", "cvar"),
    n_steps: int = 30,
    n_paths: int = 200_000,
    n_test: int = 100_000,
    epochs: int = 40,
    cvar_batch_size: int = 2048,
    cvar_epochs_factor: float = 4.5,
    cvar_cost_rates: tuple[float, ...] | None = None,
    lam: float = 1.0,
    alpha: float = 0.95,
    seed: int = 0,
    pnl_at: float = 0.002,
) -> dict:
    """Hypothesis: under costs the learned strategy attains a lower risk, by trading less.

    The delta hedge is derived where trading is free, so once it is not it is
    solving the wrong problem: it rebalances to a target that ignores the cost of
    rebalancing. A strategy trained on the true objective should hold a band
    around the delta and trade only when leaving it is worth paying for.

    The comparison is paired on identical paths, the risk difference carries a
    bootstrap standard error from the joint resample, and turnover is reported
    next to risk, so a win bought by trading more would show.

    CVaR gets a smaller batch and more epochs. Its Rockafellar-Uryasev gradient
    is proportional to the indicator that the loss exceeds w, so at the 95% level
    nineteen paths in twenty contribute nothing and each step carries less
    information. Measured on the zero-cost delta-recovery test, the deviation
    tracks the number of gradient steps and almost nothing else: CVaR needs about
    5,000 where the entropic measure converges by 3,000, and with costs the
    requirement roughly triples because the position enters the state. The
    default is 4.5x the entropic epochs at half the batch, 17,640 steps on
    200,000 paths. The first version of this study ran 5,880 and reported the
    learned strategy losing above 10 bp; that was an under-trained optimiser, not
    a finding, caught because its risk sat above a strategy the network can
    represent. The tables are in the README.
    """
    S_train = gbm_paths(n_paths, n_steps, MATURITY, SPOT, SIGMA, seed=1)
    S_test = gbm_paths(n_test, n_steps, MATURITY, SPOT, SIGMA, seed=2)
    analytic = bs_delta_positions(S_test, STRIKE, MATURITY, SIGMA)

    out: dict = {"cost_rates": list(cost_rates), "n_steps": n_steps, "by_risk": {}}
    for risk in risks:
        label = (
            f"entropic risk, $\\lambda$ = {lam:g}" if risk == "entropic" else f"CVaR at {alpha:.0%}"
        )
        is_cvar = risk == "cvar"
        risk_epochs = int(epochs * cvar_epochs_factor) if is_cvar else epochs
        risk_batch = cvar_batch_size if is_cvar else TrainConfig().batch_size
        grid = cost_rates if not is_cvar or cvar_cost_rates is None else cvar_cost_rates
        rows = {
            "delta hedge": {k: [] for k in ("risk", "risk_se", "turnover", "turnover_se")},
            "deep hedge": {k: [] for k in ("risk", "risk_se", "turnover", "turnover_se")},
        }
        differences, difference_ses, records = [], [], []
        for cost_rate in grid:
            config = TrainConfig(
                risk=risk,
                lam=lam,
                alpha=alpha,
                epochs=risk_epochs,
                batch_size=risk_batch,
                seed=seed,
            )
            result = train_hedger(S_train, STRIKE, MATURITY, cost_rate, config=config)
            learned = positions_from_hedger(result.hedger, S_test)

            deep = evaluate_strategy(S_test, learned, STRIKE, cost_rate, risk, lam, alpha)
            delta = evaluate_strategy(S_test, analytic, STRIKE, cost_rate, risk, lam, alpha)
            paired = paired_risk_difference(
                S_test, learned, analytic, STRIKE, cost_rate, risk, lam, alpha
            )
            for name, summary in [("deep hedge", deep), ("delta hedge", delta)]:
                rows[name]["risk"].append(summary["risk"])
                rows[name]["risk_se"].append(summary["risk_se"])
                rows[name]["turnover"].append(summary["mean_turnover"])
                rows[name]["turnover_se"].append(summary["mean_turnover_se"])
            differences.append(paired["difference"])
            difference_ses.append(paired["se"])
            records.append({"cost_rate": cost_rate, "deep": deep, "delta": delta, "paired": paired})

            if risk == "entropic" and np.isclose(cost_rate, pnl_at):
                import torch

                with torch.no_grad():
                    p_deep = pnl(_to_tensor(S_test), _to_tensor(learned), STRIKE, cost_rate).numpy()
                    p_delta = pnl(
                        _to_tensor(S_test), _to_tensor(analytic), STRIKE, cost_rate
                    ).numpy()
                plots.pnl_distributions(
                    p_delta, p_deep, figure_dir / "pnl_distributions.png", cost_rate, label
                )

        suffix = "entropic" if risk == "entropic" else "cvar"
        figure = plots.risk_and_turnover_against_cost(
            np.array(grid),
            {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in rows.items()},
            figure_dir / f"risk_vs_cost_{suffix}.png",
            label,
        )
        advantage = plots.risk_advantage(
            np.array(grid),
            np.array(differences),
            np.array(difference_ses),
            figure_dir / f"risk_advantage_{suffix}.png",
            label,
        )
        out["by_risk"][risk] = {
            "label": label,
            "epochs": risk_epochs,
            "batch_size": risk_batch,
            "steps": risk_epochs * -(-n_paths // risk_batch),
            "cost_rates": list(grid),
            "records": records,
            "differences": differences,
            "difference_ses": difference_ses,
            "figure": figure,
            "advantage_figure": advantage,
        }
    return out


def heston_experiment(
    params: HestonParams,
    maturity: float = 30.0 / 365.0,
    n_steps: int = 30,
    n_paths: int = 200_000,
    n_test: int = 100_000,
    epochs: int = 50,
    cost_rates: tuple[float, ...] = (0.0, 0.001, 0.005),
    risk: str = "entropic",
    lam: float = 1.0,
    alpha: float = 0.95,
    substeps: int = 4,
    seed: int = 0,
) -> dict:
    """The same question under stochastic volatility, at calibrated parameters.

    Three benchmarks rather than one. The model delta hedges the spot exposure;
    the minimum-variance delta also projects the unhedgeable variance exposure
    onto the spot, the classical correction any comparison must control for; the
    learned strategy sees the variance in its state and is trained with costs.

    If the network only recovered the minimum-variance correction it would beat
    the model delta at zero cost and gain nothing from the cost term, so the
    zero-cost column is the control.
    """
    S_train, v_train = heston_paths(
        n_paths, n_steps, maturity, params, SPOT, seed=1, substeps=substeps
    )
    S_test, v_test = heston_paths(
        n_test, n_steps, maturity, params, SPOT, seed=2, substeps=substeps
    )
    strike = SPOT  # at the money, which is where the BTC surface is liquid

    model_delta = heston_delta_positions(
        S_test, v_test, strike, maturity, params, minimum_variance=False
    )
    min_var_delta = heston_delta_positions(
        S_test, v_test, strike, maturity, params, minimum_variance=True
    )

    records = []
    for cost_rate in cost_rates:
        config = TrainConfig(risk=risk, lam=lam, alpha=alpha, epochs=epochs, seed=seed)
        result = train_hedger(S_train, strike, maturity, cost_rate, v=v_train, config=config)
        learned = positions_from_hedger(result.hedger, S_test, v_test)

        entry = {"cost_rate": cost_rate}
        for name, deltas in [
            ("deep hedge", learned),
            ("delta hedge", model_delta),
            ("minimum-variance delta", min_var_delta),
        ]:
            entry[name] = evaluate_strategy(S_test, deltas, strike, cost_rate, risk, lam, alpha)
        entry["vs model delta"] = paired_risk_difference(
            S_test, learned, model_delta, strike, cost_rate, risk, lam, alpha
        )
        entry["vs minimum-variance"] = paired_risk_difference(
            S_test, learned, min_var_delta, strike, cost_rate, risk, lam, alpha
        )
        records.append(entry)

    return {
        "params": params,
        "maturity": maturity,
        "n_steps": n_steps,
        "records": records,
        "risk": risk,
        "lam": lam,
        "alpha": alpha,
        "variance_hits_zero": float((v_test[:, 1:] == 0.0).mean()),
        "feller": params.feller(),
    }
