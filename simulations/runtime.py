"""Benchmark feature-importance runtimes as the sample size increases (Table S1).

The experiment uses the independent-covariate, linear-signal setting. Data-generation
time is not included in the reported method runtime.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods.data_generation import cnts_bin_indep

# rpy2 must use the R installation it was built against.  In a conda
# environment, prefer that environment's R unless R_HOME was set explicitly.
_CONDA_R_HOME = Path(sys.prefix) / "lib" / "R"
if "R_HOME" not in os.environ and _CONDA_R_HOME.is_dir():
    os.environ["R_HOME"] = str(_CONDA_R_HOME)

from methods.jitter import (
    _get_cforest_r_backend,
    jitterRF,
    run_cforest,
    run_ufi,
)


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
# sample sizes for Table S1 (tab:runtime_n). CForest is far slower: to reproduce its
# column, enable "CForest" in METHOD_NAMES and run with N_VALUES = (250, 500, 1000).
N_VALUES = (250, 500, 1000, 2000, 4000, 8000)
N_TRIALS = 5
N_ESTIMATORS = 100

P_CNTS = 25
P_BIN = 25
N_TRUE_CNTS = 5
N_TRUE_BIN = 5
SNR = 1
SIGNAL_TYPE = "linear"
JITTER_STRENGTH = 0.0001
RANDOM_SEED = 123

OUTPUT_DIR = Path(__file__).resolve().parent / "results_time_analysis"
METHOD_NAMES = (
    "RF",
    "RF-one-time",
    "RF-per-tree",
    "UFI",
    # "CForest",
)


def make_methods(n_estimators):
    """Return the five methods with a common number of trees."""
    rf_args = {
        "n_estimators": n_estimators,
        # Use one CPU thread for comparable timings across RF variants.
        "n_jobs": 1,
    }
    return {
        "RF": lambda X, y: jitterRF(
            X,
            y,
            task="regression",
            jitter_method=None,
            **rf_args,
        ),
        "RF-one-time": lambda X, y: jitterRF(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=JITTER_STRENGTH,
            **rf_args,
        ),
        "RF-per-tree": lambda X, y: jitterRF(
            X,
            y,
            task="regression",
            jitter_method="per_tree",
            jitter_strength=JITTER_STRENGTH,
            **rf_args,
        ),
        "UFI": lambda X, y: run_ufi(
            X,
            y,
            task="regression",
            **rf_args,
        ),
        "CForest": lambda X, y: run_cforest(
            X,
            y,
            task="regression",
            n_estimators=n_estimators,
        ),
    }


def sample_true_features(trial):
    """Use the same true features for a given trial at every sample size."""
    rng = np.random.default_rng(RANDOM_SEED + 2_000_000 + trial)
    true_cnts_idx = np.sort(
        rng.choice(P_CNTS, size=N_TRUE_CNTS, replace=False)
    )
    true_bin_idx = np.sort(
        rng.choice(P_BIN, size=N_TRUE_BIN, replace=False)
    )
    return true_cnts_idx, true_bin_idx


def generate_data(n, trial, true_cnts_idx, true_bin_idx):
    """Generate one independent-covariate, linear-signal data set."""
    return cnts_bin_indep(
        n=n,
        p_cnts=P_CNTS,
        p_bin=P_BIN,
        true_cnts_idx=true_cnts_idx,
        true_bin_idx=true_bin_idx,
        snr=SNR,
        signal_type=SIGNAL_TYPE,
        seed=data_seed(n, trial),
    )


def data_seed(n, trial):
    """Return a deterministic seed for one generated data set."""
    return RANDOM_SEED + 10_000 * n + trial


def method_seed(n_index, trial, method_index):
    """Return a seed independent of the randomized method execution order."""
    return (
        RANDOM_SEED
        + 1_000_000 * n_index
        + 10_000 * trial
        + 100 * method_index
    )


def set_cforest_seed(seed):
    """Set R's random seed after loading the CForest backend."""
    ro, _, _ = _get_cforest_r_backend()
    ro.r["set.seed"](int(seed))


def summarize_timings(timing_df):
    """Calculate trial-to-trial runtime summaries."""
    return (
        timing_df.groupby(["n", "method"], sort=False)
        .agg(
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            median_runtime_seconds=("runtime_seconds", "median"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
        )
        .reset_index()
    )


def save_runtime_plot(summary_df, output_path):
    """Plot mean runtime with one-standard-deviation error bars."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200, constrained_layout=True)

    for method_name in METHOD_NAMES:
        method_df = summary_df[summary_df["method"] == method_name]
        ax.errorbar(
            method_df["n"],
            method_df["mean_runtime_seconds"],
            yerr=method_df["std_runtime_seconds"].fillna(0),
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=method_name,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(summary_df["n"].unique()))
    ax.set_xticklabels(
        [str(value) for value in sorted(summary_df["n"].unique())]
    )
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Mean runtime (seconds, log scale)")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary(summary_df):
    """Print a compact mean (standard deviation) table."""
    display_df = summary_df.copy()
    display_df["runtime"] = display_df.apply(
        lambda row: (
            f"{row['mean_runtime_seconds']:.4f} "
            f"({row['std_runtime_seconds']:.4f})"
        ),
        axis=1,
    )
    table = display_df.pivot(index="n", columns="method", values="runtime")
    table = table.reindex(columns=METHOD_NAMES)
    print("\nRuntime in seconds: mean (standard deviation)")
    print(table.to_string())


def save_results(timing_rows, output_dir):
    """Checkpoint trial-level and summary results."""
    timing_df = pd.DataFrame(timing_rows)
    summary_df = summarize_timings(timing_df)
    timing_df.to_csv(output_dir / "runtime_by_trial.csv", index=False)
    summary_df.to_csv(output_dir / "runtime_summary.csv", index=False)
    return timing_df, summary_df


def warm_up(methods):
    """Exclude one-time Python/R initialization from measured runtimes."""
    true_cnts_idx, true_bin_idx = sample_true_features(trial=0)
    X, y = generate_data(
        n=100,
        trial=9_999,
        true_cnts_idx=true_cnts_idx,
        true_bin_idx=true_bin_idx,
    )

    print("Warming up all methods (not recorded)...", flush=True)
    for method_index, (method_name, method) in enumerate(methods.items()):
        seed = RANDOM_SEED + 90_000_000 + method_index
        np.random.seed(seed)
        if method_name == "CForest":
            set_cforest_seed(seed)
        method(X, y)
        print(f"  Warmed up {method_name}", flush=True)
    gc.collect()


def run_time_analysis(
    n_values,
    n_trials,
    n_estimators,
    output_dir,
    do_warmup=True,
):
    """Run the full timing experiment and save trial-level results."""
    methods = make_methods(n_estimators)
    timing_rows = []
    method_indices = {
        method_name: index
        for index, method_name in enumerate(METHOD_NAMES)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if do_warmup:
        warm_up(methods)

    for n_index, n in enumerate(n_values):
        for trial in range(n_trials):
            true_cnts_idx, true_bin_idx = sample_true_features(trial)
            X, y = generate_data(
                n,
                trial,
                true_cnts_idx,
                true_bin_idx,
            )
            print(f"n={n}, trial {trial + 1}/{n_trials}", flush=True)

            execution_rng = np.random.default_rng(
                RANDOM_SEED + 50_000_000 + 10_000 * n_index + trial
            )
            execution_order = list(METHOD_NAMES)
            execution_rng.shuffle(execution_order)

            for order_index, method_name in enumerate(execution_order):
                method = methods[method_name]
                method_index = method_indices[method_name]
                seed = method_seed(n_index, trial, method_index)
                # Re-seeding makes the NumPy/scikit-learn methods reproducible
                # without assigning one identical seed to every per-tree fit.
                np.random.seed(seed)
                if method_name == "CForest":
                    set_cforest_seed(seed)

                gc.collect()
                start_time = perf_counter()
                result = method(X, y)
                runtime = perf_counter() - start_time
                timing_rows.append(
                    {
                        "n": n,
                        "trial": trial + 1,
                        "method": method_name,
                        "runtime_seconds": runtime,
                        "method_reported_runtime_seconds": float(
                            result["runtime"]
                        ),
                        "data_seed": data_seed(n, trial),
                        "method_seed": seed,
                        "execution_order": order_index + 1,
                    }
                )
                print(
                    f"  {method_name:<12} {runtime:.4f} seconds",
                    flush=True,
                )

            # Checkpoint outside the timed regions so an interrupted long run
            # retains all fully completed trials.
            save_results(timing_rows, output_dir)

    timing_df, summary_df = save_results(timing_rows, output_dir)
    save_runtime_plot(summary_df, output_dir / "runtime_by_n.png")
    print_summary(summary_df)
    print(f"\nResults saved to {output_dir.resolve()}", flush=True)

    return timing_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=list(N_VALUES),
        help="sample sizes to benchmark (default: 250 500 1000 2000 4000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help="number of trials per sample size (default: 5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="number of trees used by every method (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for CSV files and the runtime plot",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="include possible first-call effects by skipping warm-up",
    )
    args = parser.parse_args()

    if any(n <= 0 for n in args.n_values):
        parser.error("all sample sizes must be positive")
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.n_estimators <= 0:
        parser.error("--n-estimators must be positive")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    run_time_analysis(
        n_values=cli_args.n_values,
        n_trials=cli_args.trials,
        n_estimators=cli_args.n_estimators,
        output_dir=cli_args.output_dir,
        do_warmup=not cli_args.skip_warmup,
    )
