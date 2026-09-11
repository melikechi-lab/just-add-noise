"""Run all four AUC scenarios and create two publication-ready 2x2 figures.

Shared settings and feature-importance methods come from ``auc_ranking_base.py``.
Run this file directly to recompute all scenarios and save the combined RF and
XGBoost figures (Table 1, Fig. S1).
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import auc_ranking_base as base
from methods.data_generation import cnts_bin_corr, cnts_bin_indep


OUTPUT_DIR = Path("results_auc_four_scenarios")
FIGURE_DPI = 300
FPR_GRID = np.linspace(0, 1, 101)

# The order determines the positions in the 2x2 figure: rows describe the
# dependence structure and columns describe the signal form.
SCENARIOS = (
    ("independent", "linear", "(a) Independent · Linear"),
    ("independent", "nonlinear", "(b) Independent · Nonlinear"),
    ("correlated", "linear", "(c) Correlated · Linear"),
    ("correlated", "nonlinear", "(d) Correlated · Nonlinear"),
)


def generate_data(
    dependence,
    signal_type,
    seed,
    true_cnts_idx,
    true_bin_idx,
):
    """Generate one data set for a dependence/signal combination."""
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


def run_scenario(dependence, signal_type, methods):
    """Run all trials for one scenario and return AUCs and ROC curves."""
    auc_rows = []
    tpr_by_method = {name: [] for name in methods}

    for trial in range(base.N_TRIALS):
        true_cnts_idx, true_bin_idx = base.sample_true_features(trial)
        X, y = generate_data(
            dependence,
            signal_type,
            base.RANDOM_SEED + trial,
            true_cnts_idx,
            true_bin_idx,
        )

        true_labels = np.zeros(base.P_CNTS + base.P_BIN, dtype=int)
        true_labels[true_cnts_idx] = 1
        true_labels[base.P_CNTS + true_bin_idx] = 1

        print(
            f"[{dependence} / {signal_type}] "
            f"Trial {trial + 1}/{base.N_TRIALS}",
            flush=True,
        )
        for method_index, (method_name, method) in enumerate(methods.items()):
            np.random.seed(
                base.RANDOM_SEED
                + 100_000 * trial
                + 1_000 * method_index
            )
            result = method(X, y)
            importances = np.asarray(result["importances"], dtype=float)

            auc_value = roc_auc_score(true_labels, importances)
            fpr, tpr, _ = roc_curve(true_labels, importances)
            grid_tpr = np.interp(FPR_GRID, fpr, tpr)
            grid_tpr[0] = 0
            tpr_by_method[method_name].append(grid_tpr)

            auc_rows.append(
                {
                    "dependence": dependence,
                    "signal_type": signal_type,
                    "trial": trial + 1,
                    "method": method_name,
                    "auc": auc_value,
                    "runtime_seconds": result["runtime"],
                }
            )

    return pd.DataFrame(auc_rows), tpr_by_method


def save_combined_roc(
    method_names,
    scenario_curves,
    output_path,
):
    """Save one clean 2x2 ROC figure with shared axes and legend."""
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
            tpr_by_method = scenario_curves[(dependence, signal_type)]
            for method_name in method_names:
                ax.plot(
                    FPR_GRID,
                    np.mean(tpr_by_method[method_name], axis=0),
                    label=method_name,
                    linewidth=2.3,
                    solid_capstyle="round",
                    zorder=3,
                    **base.ROC_STYLES.get(method_name, {}),
                )

            ax.plot(
                [0, 1],
                [0, 1],
                color="#666666",
                linestyle=(0, (5, 4)),
                linewidth=1.6,
                label="Random ranking",
                zorder=2,
            )
            ax.set_title(panel_title, pad=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(axis="both", which="major", length=4, width=1)
            ax.grid(
                axis="both",
                color="#D9D9D9",
                linewidth=0.75,
                alpha=0.65,
            )
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.supxlabel("False positive rate", fontsize=16, y=0.115)
        fig.supylabel("True positive rate", fontsize=16, x=0.025)
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=3,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#D0D0D0",
            borderpad=0.7,
            handlelength=3.0,
            columnspacing=1.8,
        )
        legend.get_frame().set_linewidth(0.8)

        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.20,
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


def save_tables(auc_df, scenario_curves):
    """Save trial-level results, summaries, and mean curves for replotting."""
    auc_df.to_csv(OUTPUT_DIR / "auc_by_trial.csv", index=False)
    summary_df = (
        auc_df.groupby(
            ["dependence", "signal_type", "method"],
            sort=False,
        )
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(OUTPUT_DIR / "auc_summary.csv", index=False)

    curve_rows = []
    for dependence, signal_type, _ in SCENARIOS:
        curves = scenario_curves[(dependence, signal_type)]
        for method_name, trial_curves in curves.items():
            mean_tpr = np.mean(trial_curves, axis=0)
            curve_rows.extend(
                {
                    "dependence": dependence,
                    "signal_type": signal_type,
                    "method": method_name,
                    "false_positive_rate": fpr,
                    "mean_true_positive_rate": tpr,
                }
                for fpr, tpr in zip(FPR_GRID, mean_tpr)
            )
    pd.DataFrame(curve_rows).to_csv(
        OUTPUT_DIR / "mean_roc_curves.csv",
        index=False,
    )


def load_saved_curves():
    """Load previously saved mean ROC curves without rerunning models."""
    curve_path = OUTPUT_DIR / "mean_roc_curves.csv"
    if not curve_path.exists():
        raise FileNotFoundError(
            f"Cannot use --plot-only because {curve_path} does not exist."
        )

    curve_df = pd.read_csv(curve_path)
    scenario_curves = {}
    for dependence, signal_type, _ in SCENARIOS:
        scenario_df = curve_df[
            (curve_df["dependence"] == dependence)
            & (curve_df["signal_type"] == signal_type)
        ]
        curves = {}
        for method_name, method_df in scenario_df.groupby("method", sort=False):
            method_df = method_df.sort_values("false_positive_rate")
            saved_fpr = method_df["false_positive_rate"].to_numpy()
            if len(saved_fpr) != len(FPR_GRID) or not np.allclose(
                saved_fpr,
                FPR_GRID,
            ):
                raise ValueError(
                    f"Unexpected FPR grid for {dependence}/{signal_type}/"
                    f"{method_name}."
                )
            curves[method_name] = [
                method_df["mean_true_positive_rate"].to_numpy()
            ]
        scenario_curves[(dependence, signal_type)] = curves

    return scenario_curves


def save_all_figures(scenario_curves):
    """Save the RF and XGBoost combined figures."""
    save_combined_roc(
        base.RF_METHOD_NAMES,
        scenario_curves,
        OUTPUT_DIR / "mean_roc_rf_four_scenarios.png",
    )
    save_combined_roc(
        base.XGB_METHOD_NAMES,
        scenario_curves,
        OUTPUT_DIR / "mean_roc_xgb_four_scenarios.png",
    )


def main(plot_only=False):
    if base.N_TRUE_CNTS + base.N_TRUE_BIN == 0:
        raise ValueError("At least one true feature is required for ROC/AUC.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if plot_only:
        scenario_curves = load_saved_curves()
        save_all_figures(scenario_curves)
        print(f"Figures saved to {OUTPUT_DIR.resolve()}", flush=True)
        return

    methods = base.make_methods()
    auc_frames = []
    scenario_curves = {}

    for dependence, signal_type, _ in SCENARIOS:
        auc_df, curves = run_scenario(dependence, signal_type, methods)
        auc_frames.append(auc_df)
        scenario_curves[(dependence, signal_type)] = curves

    all_auc_df = pd.concat(auc_frames, ignore_index=True)
    save_tables(all_auc_df, scenario_curves)
    save_all_figures(scenario_curves)
    print(f"\nResults saved to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw figures from saved mean ROC curves without simulation",
    )
    args = parser.parse_args()
    main(plot_only=args.plot_only)
