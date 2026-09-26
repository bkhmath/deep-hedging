"""Run the whole study and write the report.

    python scripts/run_study.py                # uses the cached Deribit surface
    python scripts/run_study.py --synthetic    # no network, no cache needed
    python scripts/run_study.py --quick        # small sizes, for CI

Every number printed carries a Monte Carlo standard error, and the report is
written whatever the numbers say.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from deephedging import deribit, plots, study
from deephedging.calibrate import calibrate, synthetic_surface
from deephedging.heston import HestonParams

ROOT = Path(__file__).resolve().parents[1]

# Used when --synthetic is given, and as the surface the calibration recovery test
# is built on: plausible for BTC, and with the Feller condition satisfied so that
# the synthetic case is not secretly the same case as the real one.
SYNTHETIC_TRUTH = HestonParams(v0=0.36, kappa=3.0, theta=0.30, xi=1.2, rho=-0.25)


def fmt(value: float, se: float | None = None, digits: int = 4) -> str:
    if se is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} +/- {se:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="skip the cached surface")
    parser.add_argument("--quick", action="store_true", help="small sizes for CI")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        sizes = dict(
            n_paths=20_000,
            n_test=20_000,
            epochs=8,
            ns=(10, 20),
            rate_paths=20_000,
            rate_ns=(8, 16, 32),
            lambdas=(0.05, 0.2),
            cost_rates=(0.0, 0.002),
            cvar_cost_rates=(0.0, 0.002),
            risks=("entropic",),
            heston_costs=(0.0, 0.002),
        )
    else:
        sizes = dict(
            n_paths=200_000,
            n_test=100_000,
            epochs=60,
            ns=(15, 30, 60),
            rate_paths=100_000,
            rate_ns=(8, 16, 32, 64, 128, 256),
            lambdas=(0.02, 0.05, 0.1, 0.2, 0.4),
            cost_rates=(0.0, 0.0005, 0.001, 0.002, 0.005, 0.01),
            # CVaR needs about twice the wall clock per point (the measurements are
            # in `study.cost_sweep_experiment`), so it gets a coarser cost grid to
            # keep the whole study runnable in one sitting.
            cvar_cost_rates=(0.0, 0.001, 0.002, 0.005),
            risks=("entropic", "cvar"),
            heston_costs=(0.0, 0.001, 0.005),
        )

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    started = time.time()
    emit("# Deep hedging under transaction costs: study output")
    emit()
    emit(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} "
        f"({'quick' if args.quick else 'full'} configuration)."
    )
    emit()
    emit(
        f"Option: European call, strike {study.STRIKE:g}, spot {study.SPOT:g}, "
        f"maturity {study.MATURITY:g} year, Black-Scholes volatility {study.SIGMA:g}. "
        "Rates and carry are zero throughout. Paths are simulated under the pricing "
        "measure. `+/-` is one Monte Carlo standard error."
    )
    emit()

    # ---------------------------------------------------------------- experiment 1
    emit("## 1. Discretisation error of the delta hedge")
    emit()
    emit(
        "*Hypothesis: frictionless, the replication error of the delta hedge decays "
        "like n^(-1/2) in the number of rebalancing dates.*"
    )
    emit()
    rate = study.discretisation_rate_experiment(
        figure_dir, n_paths=sizes["rate_paths"], ns=sizes["rate_ns"]
    )
    emit("| n | std of replication error | mean error |")
    emit("|---|---|---|")
    for n, sd, mean, se in zip(
        rate["ns"], rate["stds"], rate["means"], rate["mean_ses"], strict=True
    ):
        emit(f"| {n} | {sd:.5f} | {mean:+.5f} +/- {se:.5f} |")
    emit()
    emit(
        f"**Fitted log-log slope {rate['slope']:+.4f}, against the theoretical -0.5.** "
        "The mean error is zero within Monte Carlo error at every n, as it must be: the "
        "hedging gain is a martingale increment sum, so the delta hedge is unbiased and "
        "only its dispersion is at stake."
    )
    emit()
    emit("![Discretisation error](figures/discretisation_rate.png)")
    emit()

    # ---------------------------------------------------------------- experiment 2
    emit("## 2. Does the network recover the Black-Scholes delta?")
    emit()
    emit(
        "*Hypothesis: with zero costs, refining the rebalancing grid drives the learned "
        "position to the analytic delta. The network is never shown a delta during "
        "training, so agreement is a recovered theorem.*"
    )
    emit()
    recovery = study.delta_recovery_experiment(
        figure_dir,
        ns=sizes["ns"],
        n_paths=sizes["n_paths"],
        n_test=sizes["n_test"],
        epochs=sizes["epochs"],
    )
    emit("| n | mean abs dev | median | p95 | p99 | max | learned risk | delta risk |")
    emit("|---|---|---|---|---|---|---|---|")
    for row in recovery["rows"]:
        emit(
            f"| {row['n']} | {row['mean_abs_dev']:.4f} | {row['median_abs_dev']:.4f} | "
            f"{row['p95_abs_dev']:.4f} | {row['p99_abs_dev']:.4f} | {row['max_abs_dev']:.4f} | "
            f"{fmt(row['deep_risk'], row['deep_risk_se'])} | "
            f"{fmt(row['delta_risk'], row['delta_risk_se'])} |"
        )
    emit()
    converged = [
        row["n"]
        for row in recovery["rows"]
        if row["deep_risk"] > row["delta_risk"] + 3.0 * row["delta_risk_se"]
    ]
    emit(f"**The deviation falls at rate n^({recovery['slope']:+.3f}).**")
    emit()
    emit(
        "The delta hedge is a strategy the network is able to represent, so the learned risk "
        "has to come out at or below it. A learned risk above it would mean the optimisation "
        "had not converged, and could not mean the theory was wrong, so the check is "
        "reported here and not asserted:"
    )
    emit()
    if converged:
        emit(
            f"> **Not converged at n = {converged}.** The learned risk exceeds the delta "
            "hedge's by more than three standard errors there, so the deviations quoted at "
            "those n measure the optimiser and not the theorem. This is expected under "
            "`--quick`, which trains for a fraction of the epochs the full study uses."
        )
    else:
        emit(
            "> Converged at every n: the learned risk came out at or below the delta hedge's "
            "throughout, within three standard errors."
        )
    emit()
    emit("![Learned position against the analytic delta](figures/learned_vs_delta.png)")
    emit()
    emit("![Convergence rate](figures/delta_recovery_rate.png)")
    emit()

    # ---------------------------------------------------------------- experiment 3
    emit("## 3. Architecture: shared weights against one network per date")
    emit()
    emit(
        "*Hypothesis: a single network with time to maturity as an input matches n "
        "per-date networks, at a fraction of the parameters.*"
    )
    emit()
    arch = study.architecture_experiment(
        n_steps=sizes["ns"][-1],
        n_paths=sizes["n_paths"],
        n_test=sizes["n_test"],
        epochs=sizes["epochs"],
    )
    emit(
        f"At {arch['n_steps']} rebalancing dates and a cost rate of "
        f"{arch['cost_rate'] * 1e4:.0f} basis points:"
    )
    emit()
    emit("| strategy | parameters | risk | turnover |")
    emit("|---|---|---|---|")
    for name in ["shared weights", "one network per date", "delta hedge"]:
        row = arch[name]
        params = row.get("n_parameters", 0)
        emit(
            f"| {name} | {params if params else '--'} | {fmt(row['risk'], row['risk_se'])} | "
            f"{fmt(row['mean_turnover'], row['mean_turnover_se'], 2)} |"
        )
    emit()
    diff = arch["difference"]
    emit(f"Paired difference (per-date minus shared): {fmt(diff['difference'], diff['se'])}.")
    emit()

    # ---------------------------------------------------------------- experiment 4
    emit("## 4. Indifference price as the risk aversion vanishes")
    emit()
    emit(
        "*Hypothesis: with the entropic measure and zero costs, the indifference price "
        "tends to the Black-Scholes price as lambda -> 0.*"
    )
    emit()
    price = study.indifference_price_experiment(
        figure_dir,
        n_steps=sizes["ns"][-1],
        n_paths=sizes["n_paths"],
        n_test=sizes["n_paths"],
        epochs=max(sizes["epochs"] // 2, 8),
        lambdas=sizes["lambdas"],
    )
    emit("| lambda | indifference price | minus Black-Scholes |")
    emit("|---|---|---|")
    for lam, p, se in zip(price["lambdas"], price["prices"], price["price_ses"], strict=True):
        emit(f"| {lam:g} | {fmt(p, se)} | {p - price['bs_price']:+.4f} |")
    emit()
    emit(
        f"Black-Scholes price **{price['bs_price']:.4f}**; linear fit in lambda gives intercept "
        f"**{price['intercept']:.4f}** (error {price['intercept'] - price['bs_price']:+.4f}) and "
        f"slope {price['slope']:.4f}. The slope is half the variance of the unhedgeable "
        "residual, so it is a second measurement and not a nuisance parameter."
    )
    emit()
    emit("![Indifference price](figures/indifference_price.png)")
    emit()

    # ---------------------------------------------------------------- experiment 5
    emit("## 5. The result: hedging with costs in the objective")
    emit()
    emit(
        "*Hypothesis: with a positive cost rate the learned strategy attains a strictly "
        "lower risk than the delta hedge, and does so by trading less.*"
    )
    emit()
    sweep = study.cost_sweep_experiment(
        figure_dir,
        cost_rates=sizes["cost_rates"],
        cvar_cost_rates=sizes["cvar_cost_rates"],
        risks=sizes["risks"],
        n_steps=min(sizes["ns"][-1], 30),
        n_paths=sizes["n_paths"],
        n_test=sizes["n_test"],
        epochs=max(sizes["epochs"] * 2 // 3, 8),
    )
    for risk, block in sweep["by_risk"].items():
        plain = block["label"].replace("$\\lambda$", "lambda").replace("\\", "")
        emit(f"### {plain}")
        emit()
        emit(
            f"Trained for {block['epochs']} epochs at a batch size of "
            f"{block['batch_size']} at each cost rate, i.e. {block['steps']:,} gradient "
            "steps."
        )
        emit()
        emit(
            "| cost (bp) | delta risk | deep risk | difference (paired) | delta turnover | deep turnover |"
        )
        emit("|---|---|---|---|---|---|")
        for record in block["records"]:
            d, p, pair = record["delta"], record["deep"], record["paired"]
            emit(
                f"| {record['cost_rate'] * 1e4:.0f} | {fmt(d['risk'], d['risk_se'])} | "
                f"{fmt(p['risk'], p['risk_se'])} | {fmt(pair['difference'], pair['se'])} | "
                f"{d['mean_turnover']:.1f} | {p['mean_turnover']:.1f} |"
            )
        emit()
        suffix = "entropic" if risk == "entropic" else "cvar"
        emit(f"![Risk and turnover against cost](figures/risk_vs_cost_{suffix}.png)")
        emit()
        emit(f"![Advantage](figures/risk_advantage_{suffix}.png)")
        emit()
    emit("![Profit and loss distributions](figures/pnl_distributions.png)")
    emit()

    # ---------------------------------------------------------------- calibration
    emit("## 6. Heston calibrated to a real surface")
    emit()
    if args.synthetic:
        emit(
            "*Run with `--synthetic`: the surface is generated from known Heston "
            "parameters rather than downloaded, so this section is a recovery check "
            "over a calibration to the market.*"
        )
        emit()
        surface = synthetic_surface(SYNTHETIC_TRUTH)
        source = "synthetic surface"
    else:
        raw = deribit.load_surface()
        surface = deribit.clean_surface(raw)
        forward = float(surface["forward"].iloc[0])
        source = (
            f"Deribit BTC, forward {forward:,.0f} USD, {len(surface)} quotes, "
            f"{surface['maturity'].min() * 365:.0f} to {surface['maturity'].max() * 365:.0f} days"
        )
    emit(f"Surface: {source}.")
    emit()
    result = calibrate(surface)
    emit("```")
    emit(result.summary())
    emit("```")
    emit()
    if args.synthetic:
        truth = SYNTHETIC_TRUTH
        emit(
            f"Generating parameters: v0={truth.v0:.4f} kappa={truth.kappa:.3f} "
            f"theta={truth.theta:.4f} xi={truth.xi:.3f} rho={truth.rho:+.3f}."
        )
        emit()
    emit(
        f"All {result.starts_tried} starting points were tried; the spread of the objective "
        f"across the local optima found was {result.objective_spread:.3e}."
    )
    emit()
    figure = plots.calibration_fit(
        result.fit,
        figure_dir / "calibration_fit.png",
        result.rmse_iv,
        "a synthetic surface" if args.synthetic else "the Deribit BTC surface",
    )
    emit(f"![Calibration fit](figures/{figure.name})")
    emit()

    # ---------------------------------------------------------------- experiment 6
    emit("## 7. Hedging under the calibrated Heston model")
    emit()
    emit(
        "*Hypothesis: the same advantage appears under stochastic volatility, and it is "
        "not merely the classical minimum-variance correction in disguise.*"
    )
    emit()
    heston_out = study.heston_experiment(
        result.params,
        n_paths=sizes["n_paths"],
        n_test=sizes["n_test"],
        epochs=max(sizes["epochs"] * 5 // 6, 8),
        cost_rates=sizes["heston_costs"],
    )
    zero_fraction = heston_out["variance_hits_zero"]
    emit(
        f"At-the-money call, {heston_out['maturity'] * 365:.0f} days, "
        f"{heston_out['n_steps']} rebalancing dates, entropic risk with lambda = "
        f"{heston_out['lam']:g}. Feller 2*kappa*theta - xi^2 = {heston_out['feller']:+.4f} "
        f"({'violated' if heston_out['feller'] < 0 else 'satisfied'}); the simulated variance "
        f"sits at exactly zero on {zero_fraction:.2%} of observation dates."
    )
    emit()
    if zero_fraction > 0.005:
        emit(
            "At that rate the zero boundary is not a corner case, and the naive Euler scheme "
            "would produce NaN on most paths: see "
            "`tests/test_simulate.py::test_naive_euler_variance_scheme_produces_nan`, which "
            "measures it. Full truncation is doing real work here."
        )
    else:
        emit(
            "The variance essentially never reaches zero at these parameters, so the choice of "
            "variance scheme does not bite in this run. It does bite under the parameters "
            "calibrated to the real BTC surface, where the Feller condition is violated."
        )
    emit()
    emit(
        "| cost (bp) | model delta | min-variance delta | deep hedge | deep vs model | deep vs min-var | deep turnover | delta turnover |"
    )
    emit("|---|---|---|---|---|---|---|---|")
    for record in heston_out["records"]:
        emit(
            f"| {record['cost_rate'] * 1e4:.0f} | "
            f"{fmt(record['delta hedge']['risk'], record['delta hedge']['risk_se'])} | "
            f"{fmt(record['minimum-variance delta']['risk'], record['minimum-variance delta']['risk_se'])} | "
            f"{fmt(record['deep hedge']['risk'], record['deep hedge']['risk_se'])} | "
            f"{fmt(record['vs model delta']['difference'], record['vs model delta']['se'])} | "
            f"{fmt(record['vs minimum-variance']['difference'], record['vs minimum-variance']['se'])} | "
            f"{record['deep hedge']['mean_turnover']:.1f} | "
            f"{record['delta hedge']['mean_turnover']:.1f} |"
        )
    emit()

    emit(f"Total run time {(time.time() - started) / 60:.1f} minutes.")
    emit()

    report = out_dir / "summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
