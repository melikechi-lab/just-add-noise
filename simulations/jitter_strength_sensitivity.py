"""Sensitivity analysis for the magnitude of one-time jitter.

The experiment evaluates one-time-jittered random forest and default-depth
XGBoost in the four feature-recovery settings used by
``auc_ranking.py``:

    independent/correlated covariates x linear/nonlinear signal.

For each scenario and trial, every method/epsilon combination uses the same
data set.  The underlying uniform jitter realization is also paired across
epsilon values, so changing epsilon only rescales an otherwise identical
perturbation.  Here epsilon is the half-width of Uniform(-epsilon, epsilon).
Performance is measured by ROC-AUC for recovering the true features from the
estimated feature importances.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import auc_ranking_base as base
from methods.data_generation import cnts_bin_corr, cnts_bin_indep
from methods.jitter import jitterRF, jitterXGB


OUTPUT_DIR = Path(__file__).resolve().parent / (
    "results_jitter_strength_sensitivity"
)
FIGURE_DPI = 300
EPSILONS = (0.1, 0.01, 0.0001, 0.000001)
METHOD_NAMES = ("RF-one-time", "XGB-one-time")

# Rows describe dependence and columns describe signal form in the 2x2 plot.
SCENARIOS = (
    ("independent", "linear", "(a) Independent · Linear"),
    ("independent", "nonlinear", "(b) Independent · Nonlinear"),
    ("correlated", "linear", "(c) Correlated · Linear"),
    ("correlated", "nonlinear", "(d) Correlated · Nonlinear"),
)

METHOD_STYLES = {
    "RF-one-time": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "XGB-one-time": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}


def generate_data(
    dependence,
    signal_type,
    seed,
    true_cnts_idx,
    true_bin_idx,
):
    """Generate one of the four original AUC simulation settings."""
    params = dict(
        n=base.N,
        p_cnts=base.P_CNTS,
        p_bin=base.P_BIN,
        true_cnts_idx=true_cnts_idx,
        true_bin_idx=true_bin_idx,
        snr=base.SNR,
        signal_type=signal_type,
        seed=seed,
    )

    if dependence == "independent":
        return cnts_bin_indep(**params)
    if dependence == "correlated":
        return cnts_bin_corr(
            p_corr=base.P_CORR,
            target_rho=1 / np.sqrt(2),
            **params,
        )
    raise ValueError(f"Unknown dependence structure: {dependence}")


def fit_method(method_name, X, y, epsilon):
    """Fit one of the two one-time-jittered models."""
    if method_name == "RF-one-time":
        return jitterRF(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=epsilon,
            **base.RF_ARGS,
        )
    if method_name == "XGB-one-time":
        return jitterXGB(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=epsilon,
            max_depth=None,
            cat_idx=list(
                range(base.P_CNTS, base.P_CNTS + base.P_BIN)
            ),
        )
    raise ValueError(f"Unknown method: {method_name}")


def make_true_labels(true_cnts_idx, true_bin_idx):
    """Construct binary feature-relevance labels for ROC-AUC."""
    labels = np.zeros(base.P_CNTS + base.P_BIN, dtype=int)
    labels[true_cnts_idx] = 1
    labels[base.P_CNTS + true_bin_idx] = 1
    return labels


def paired_random_seed(scenario_index, trial):
    """Seed the same base jitter and model randomness at every epsilon."""
    return (
        base.RANDOM_SEED
        + 10_000_000 * scenario_index
        + 100_000 * trial
    )


def summarize_results(auc_df, epsilons):
    """Summarize AUC and runtime across trials in canonical order."""
    summary_df = (
        auc_df.groupby(
            ["dependence", "signal_type", "method", "epsilon"],
            sort=False,
        )
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            n_trials=("auc", "count"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
        )
        .reset_index()
    )
    summary_df["se_auc"] = (
        summary_df["std_auc"] / np.sqrt(summary_df["n_trials"])
    )

    scenario_order = {
        (dependence, signal_type): index
        for index, (dependence, signal_type, _) in enumerate(SCENARIOS)
    }
    method_order = {
        method_name: index
        for index, method_name in enumerate(METHOD_NAMES)
    }
    epsilon_order = {
        epsilon: index
        for index, epsilon in enumerate(epsilons)
    }
    summary_df["_scenario_order"] = [
        scenario_order[(dependence, signal_type)]
        for dependence, signal_type in zip(
            summary_df["dependence"],
            summary_df["signal_type"],
        )
    ]
    summary_df["_method_order"] = summary_df["method"].map(method_order)
    summary_df["_epsilon_order"] = summary_df["epsilon"].map(
        epsilon_order
    )
    summary_df = summary_df.sort_values(
        ["_scenario_order", "_method_order", "_epsilon_order"]
    ).drop(
        columns=[
            "_scenario_order",
            "_method_order",
            "_epsilon_order",
        ]
    )
    return summary_df.reset_index(drop=True)


def epsilon_tick_label(epsilon):
    """Format power-of-ten jitter strengths for a plot axis."""
    exponent = int(round(np.log10(epsilon)))
    if np.isclose(epsilon, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"
    return f"{epsilon:g}"


def save_sensitivity_figure(summary_df, output_path):
    """Save a publication-ready 2x2 mean-AUC sensitivity figure."""
    plot_settings = {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "savefig.dpi": FIGURE_DPI,
    }
    epsilon_ticks = sorted(summary_df["epsilon"].unique())

    with plt.rc_context(plot_settings):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12.5, 9.2),
            dpi=FIGURE_DPI,
            sharex=True,
            sharey=True,
        )

        for ax, (dependence, signal_type, panel_title) in zip(
            axes.flat,
            SCENARIOS,
        ):
            scenario_df = summary_df[
                (summary_df["dependence"] == dependence)
                & (summary_df["signal_type"] == signal_type)
            ]
            for method_name in METHOD_NAMES:
                method_df = scenario_df[
                    scenario_df["method"] == method_name
                ].sort_values("epsilon")
                ax.errorbar(
                    method_df["epsilon"],
                    method_df["mean_auc"],
                    yerr=method_df["std_auc"].fillna(0),
                    label=method_name,
                    linewidth=2.2,
                    markersize=6,
                    capsize=3,
                    zorder=3,
                    **METHOD_STYLES[method_name],
                )

            ax.axhline(
                0.5,
                color="#666666",
                linestyle=(0, (5, 4)),
                linewidth=1.4,
                zorder=2,
            )
            ax.set_title(panel_title, pad=10)
            ax.set_xscale("log")
            ax.set_xlim(min(epsilon_ticks) / 2, max(epsilon_ticks) * 2)
            ax.set_ylim(0.5, 1.0)
            ax.set_xticks(epsilon_ticks)
            ax.set_xticklabels(
                [epsilon_tick_label(value) for value in epsilon_ticks]
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(
                axis="both",
                color="#D9D9D9",
                linewidth=0.75,
                alpha=0.65,
            )
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # plain line handles so the legend shows lines, not error bars
        handles = [
            Line2D([0], [0], linewidth=2.2, markersize=6, **METHOD_STYLES[name])
            for name in METHOD_NAMES
        ]
        labels = list(METHOD_NAMES)
        fig.supxlabel(r"Jitter strength ($\delta$)", fontsize=16, y=0.08)
        fig.supylabel("Mean feature-recovery AUC", fontsize=16, x=0.025)
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(METHOD_NAMES),
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#D0D0D0",
            borderpad=0.7,
            handlelength=3.0,
            columnspacing=2.0,
        )
        legend.get_frame().set_linewidth(0.8)
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.16,
            top=0.96,
            wspace=0.12,
            hspace=0.18,
        )

        save_args = {
            "dpi": FIGURE_DPI,
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        fig.savefig(output_path, **save_args)
        fig.savefig(output_path.with_suffix(".pdf"), **save_args)
        plt.close(fig)


def save_checkpoint(auc_rows, output_dir):
    """Save all fully completed method/epsilon evaluations."""
    pd.DataFrame(auc_rows).to_csv(
        output_dir / "auc_by_trial.csv",
        index=False,
    )


def run_sensitivity_analysis(n_trials, epsilons, output_dir):
    """Run all scenarios and save trial-level and summary results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    auc_rows = []
    true_feature_rows = []

    for trial in range(n_trials):
        true_cnts_idx, true_bin_idx = base.sample_true_features(trial)
        true_feature_rows.append(
            {
                "trial": trial + 1,
                "true_continuous_indices": ",".join(
                    map(str, true_cnts_idx)
                ),
                "true_binary_indices": ",".join(map(str, true_bin_idx)),
            }
        )
    pd.DataFrame(true_feature_rows).to_csv(
        output_dir / "true_features_by_trial.csv",
        index=False,
    )

    for scenario_index, (dependence, signal_type, _) in enumerate(SCENARIOS):
        for trial in range(n_trials):
            true_cnts_idx, true_bin_idx = base.sample_true_features(trial)
            data_seed = base.RANDOM_SEED + trial
            X, y = generate_data(
                dependence,
                signal_type,
                data_seed,
                true_cnts_idx,
                true_bin_idx,
            )
            true_labels = make_true_labels(true_cnts_idx, true_bin_idx)
            random_seed = paired_random_seed(scenario_index, trial)

            print(
                f"[{dependence} / {signal_type}] "
                f"Trial {trial + 1}/{n_trials}",
                flush=True,
            )
            for method_name in METHOD_NAMES:
                for epsilon in epsilons:
                    # Resetting to the same seed pairs the uniform jitter and,
                    # for RF, the forest randomness across epsilon values.
                    np.random.seed(random_seed)
                    result = fit_method(method_name, X, y, epsilon)
                    importances = np.asarray(
                        result["importances"],
                        dtype=float,
                    )
                    if importances.shape != true_labels.shape:
                        raise ValueError(
                            f"{method_name} returned {importances.shape}; "
                            f"expected {true_labels.shape}."
                        )
                    if not np.all(np.isfinite(importances)):
                        raise ValueError(
                            f"{method_name} returned non-finite importances."
                        )
                    auc_value = roc_auc_score(true_labels, importances)

                    auc_rows.append(
                        {
                            "dependence": dependence,
                            "signal_type": signal_type,
                            "trial": trial + 1,
                            "method": method_name,
                            "epsilon": epsilon,
                            "auc": auc_value,
                            "runtime_seconds": float(result["runtime"]),
                            "data_seed": data_seed,
                            "paired_random_seed": random_seed,
                        }
                    )
                    print(
                        f"  {method_name:<13} "
                        f"epsilon={epsilon:g}: AUC={auc_value:.4f}",
                        flush=True,
                    )

            save_checkpoint(auc_rows, output_dir)

    auc_df = pd.DataFrame(auc_rows)
    summary_df = summarize_results(auc_df, epsilons)
    auc_df.to_csv(output_dir / "auc_by_trial.csv", index=False)
    summary_df.to_csv(output_dir / "auc_summary.csv", index=False)
    save_sensitivity_figure(
        summary_df,
        output_dir / "auc_by_jitter_strength.png",
    )

    print("\nAUC sensitivity summary: mean (standard deviation)")
    display_df = summary_df.copy()
    display_df["auc_mean_sd"] = display_df.apply(
        lambda row: f"{row['mean_auc']:.4f} ({row['std_auc']:.4f})",
        axis=1,
    )
    print(
        display_df[
            [
                "dependence",
                "signal_type",
                "method",
                "epsilon",
                "auc_mean_sd",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to {output_dir.resolve()}", flush=True)
    return auc_df, summary_df


def load_and_plot(output_dir):
    """Redraw the sensitivity figure from an existing summary CSV."""
    summary_path = output_dir / "auc_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot use --plot-only because {summary_path} does not exist."
        )
    summary_df = pd.read_csv(summary_path)
    save_sensitivity_figure(
        summary_df,
        output_dir / "auc_by_jitter_strength.png",
    )
    print(f"Figures saved to {output_dir.resolve()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=base.N_TRIALS,
        help=f"number of trials per scenario (default: {base.N_TRIALS})",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=list(EPSILONS),
        help="jitter strengths (default: 0.1 0.01 0.0001 0.000001)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for CSV files and the sensitivity figure",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure from a previously saved summary CSV",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if len(set(args.epsilons)) != len(args.epsilons):
        parser.error("--epsilons must not contain duplicates")
    if any(epsilon <= 0 for epsilon in args.epsilons):
        parser.error("all jitter strengths must be positive")
    args.epsilons = tuple(args.epsilons)
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.plot_only:
        load_and_plot(cli_args.output_dir)
    else:
        run_sensitivity_analysis(
            n_trials=cli_args.trials,
            epsilons=cli_args.epsilons,
            output_dir=cli_args.output_dir,
        )
