# IPSS variable-selection simulation
"""Run the IPSS feature-selection experiment behind Figure 3 (fig:ipss_all).

Writes per-scenario metric pickles to ``results/<scenario>_<signal>/``; run once per
signal type (set ``SIGNAL_TYPE`` to 'linear' then 'nonlinear'), then build the figure
with ``plot_ipss_simulation.py``.
"""

import math
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from methods.data_generation import cnts_bin_corr, cnts_bin_indep
from ipss import ipss


# ----------------------------------------------------------------
# Output settings
# ----------------------------------------------------------------
save_data = True   # write the .pkl metric files (consumed by plot_ipss_simulation.py)
save_fig = False   # write the per-run result_*.png diagnostic figures
plot_fig = False   # display the diagnostic figures interactively

if not plot_fig:
    matplotlib.use("Agg")


# ----------------------------------------------------------------
# Shared simulation settings
# ----------------------------------------------------------------
P_CNTS = 25
P_BIN = 25
P_CORR = 25
N_TRUE_CNTS = 5
N_TRUE_BIN = 5
N = 200
SNR = 1
SIGNAL_TYPE = "linear"  # "linear" or "nonlinear"
N_TRIALS = 100
RANDOM_SEED = 123
ALPHAS = np.round(np.arange(0.0, 0.5001, 0.01), 2)
SCENARIOS = ("independent", "correlated")
OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
JITTER_STRENGTH = 0.0001

METHODS = {
    "GB": {"selector": "gb", "jitter": False, "delta": None},
    "RF": {"selector": "rf", "jitter": False, "delta": None},
    "GB-jitter": {"selector": "gb", "jitter": True, "delta": 2},
    "RF-jitter": {"selector": "rf", "jitter": True, "delta": 2},
    "UFI": {"selector": "ufi", "jitter": False, "delta": 2},
}

CONT_COLS = set(range(P_CNTS))
BIN_COLS = set(range(P_CNTS, P_CNTS + P_BIN))


def jitter_binary_features(X, seed):
    """Add one fixed jitter realization to the binary feature block."""
    X_jitter = X.astype(float, copy=True)
    rng = np.random.default_rng(seed)
    X_jitter[:, P_CNTS:P_CNTS + P_BIN] += rng.uniform(
        -JITTER_STRENGTH,
        JITTER_STRENGTH,
        size=(X.shape[0], P_BIN),
    )
    return X_jitter


def sample_true_features(trial):
    """Randomly choose the true continuous and binary features for a trial."""
    rng = np.random.default_rng(RANDOM_SEED + 2_000_000 + trial)
    true_cnts_idx = np.sort(
        rng.choice(P_CNTS, size=N_TRUE_CNTS, replace=False)
    )
    true_bin_idx = np.sort(
        rng.choice(P_BIN, size=N_TRUE_BIN, replace=False)
    )
    return true_cnts_idx, true_bin_idx


def generate_data(scenario, seed, true_cnts_idx, true_bin_idx):
    params = dict(
        n=N,
        p_cnts=P_CNTS,
        p_bin=P_BIN,
        true_cnts_idx=true_cnts_idx,
        true_bin_idx=true_bin_idx,
        snr=SNR,
        signal_type=SIGNAL_TYPE,
        seed=seed,
    )

    if scenario == "independent":
        return cnts_bin_indep(**params)
    if scenario == "correlated":
        return cnts_bin_corr(
            p_corr=P_CORR,
            target_rho=1 / math.sqrt(2),
            **params,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def allocate_metrics():
    shape = (N_TRIALS, len(METHODS), len(ALPHAS))
    return {
        name: np.zeros(shape, dtype=int)
        for name in (
            "TP_all", "FP_all", "NS_all",
            "TP_cont", "FP_cont", "NS_cont",
            "TP_bin", "FP_bin", "NS_bin",
        )
    }


def record_selection(
    metrics,
    selected_per_alpha,
    trial,
    method_index,
    true_cont,
    true_bin,
):
    true_all = true_cont | true_bin

    for alpha_index in range(len(ALPHAS)):
        selected = selected_per_alpha[alpha_index]

        selected_cont = selected & CONT_COLS
        selected_bin = selected & BIN_COLS

        metrics["TP_all"][trial, method_index, alpha_index] = len(
            selected & true_all
        )
        metrics["FP_all"][trial, method_index, alpha_index] = len(
            selected - true_all
        )
        metrics["NS_all"][trial, method_index, alpha_index] = len(selected)

        metrics["TP_cont"][trial, method_index, alpha_index] = len(
            selected_cont & true_cont
        )
        metrics["FP_cont"][trial, method_index, alpha_index] = len(
            selected_cont - true_cont
        )
        metrics["NS_cont"][trial, method_index, alpha_index] = len(
            selected_cont
        )

        metrics["TP_bin"][trial, method_index, alpha_index] = len(
            selected_bin & true_bin
        )
        metrics["FP_bin"][trial, method_index, alpha_index] = len(
            selected_bin - true_bin
        )
        metrics["NS_bin"][trial, method_index, alpha_index] = len(
            selected_bin
        )


def resolve_output_dir(base):
    """Return `base`, or `base-2`, `base-3`, ... if a non-empty dir already
    exists, so previously saved results are never overwritten."""
    candidate = base
    index = 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = base.with_name(f"{base.name}-{index}")
        index += 1
    return candidate


def save_results(metrics, true_features_by_trial, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    method_names = list(METHODS)

    for metric_name, values in metrics.items():
        result = {
            method_name: values[:, method_index, :]
            for method_index, method_name in enumerate(method_names)
        }
        with open(output_dir / f"{metric_name}.pkl", "wb") as file:
            pickle.dump(result, file)

    with open(output_dir / "alphas.pkl", "wb") as file:
        pickle.dump(ALPHAS, file)

    with open(output_dir / "true_features_by_trial.pkl", "wb") as file:
        pickle.dump(true_features_by_trial, file)


def draw_curves(ax, mean_curve, ylabel, add_reference=True):
    for method_index, method_name in enumerate(METHODS):
        ax.plot(
            ALPHAS,
            mean_curve[method_index],
            label=method_name,
            linewidth=2,
        )
    if add_reference:
        ax.plot(ALPHAS, ALPHAS, "k--", linewidth=2, label="y=x")
    ax.set_xlabel("Target FDR", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.legend()


def plot_curve(mean_curve, output_path, ylabel, add_reference=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_curves(ax, mean_curve, ylabel, add_reference)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def show_curve_pair(mean_fdr, mean_tpr, category_label):
    """Display FDR and TPR side by side for one feature category."""
    fig, (fdr_ax, tpr_ax) = plt.subplots(1, 2, figsize=(14, 6))
    draw_curves(fdr_ax, mean_fdr, "FDR", add_reference=True)
    draw_curves(tpr_ax, mean_tpr, "TPR", add_reference=False)
    fig.suptitle(category_label, fontsize=20)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def save_plots(metrics, output_dir):
    mean_fdr_all = (
        metrics["FP_all"] / np.maximum(metrics["NS_all"], 1)
    ).mean(axis=0)
    mean_tpr_all = (
        metrics["TP_all"] / (N_TRUE_CNTS + N_TRUE_BIN)
    ).mean(axis=0)

    mean_fdr_cont = (
        metrics["FP_cont"] / np.maximum(metrics["NS_cont"], 1)
    ).mean(axis=0)
    mean_tpr_cont = (metrics["TP_cont"] / N_TRUE_CNTS).mean(axis=0)

    mean_fdr_bin = (
        metrics["FP_bin"] / np.maximum(metrics["NS_bin"], 1)
    ).mean(axis=0)
    mean_tpr_bin = (metrics["TP_bin"] / N_TRUE_BIN).mean(axis=0)

    if plot_fig:
        show_curve_pair(mean_fdr_all, mean_tpr_all, "All features")
        show_curve_pair(
            mean_fdr_cont, mean_tpr_cont, "Continuous features only"
        )
        show_curve_pair(mean_fdr_bin, mean_tpr_bin, "Binary features only")

    if not save_fig:
        return

    def path(name):
        return output_dir / name

    plot_curve(mean_fdr_all, path("result_fdr_all.png"), "FDR")
    plot_curve(
        mean_tpr_all,
        path("result_tpr_all.png"),
        "TPR",
        add_reference=False,
    )
    plot_curve(
        mean_fdr_cont,
        path("result_fdr_cont.png"),
        "FDR (continuous)",
    )
    plot_curve(
        mean_tpr_cont,
        path("result_tpr_cont.png"),
        "TPR (continuous)",
        add_reference=False,
    )
    plot_curve(
        mean_fdr_bin,
        path("result_fdr_bin.png"),
        "FDR (binary)",
    )
    plot_curve(
        mean_tpr_bin,
        path("result_tpr_bin.png"),
        "TPR (binary)",
        add_reference=False,
    )


def run_scenario(scenario):
    metrics = allocate_metrics()
    true_features_by_trial = []
    print(f"\nRunning {scenario} scenario", flush=True)

    for trial in range(N_TRIALS):
        true_cnts_idx, true_bin_idx = sample_true_features(trial)
        true_cont = set(true_cnts_idx.tolist())
        true_bin = set((P_CNTS + true_bin_idx).tolist())
        true_features_by_trial.append(
            {
                "continuous": true_cnts_idx,
                "binary": true_bin_idx,
            }
        )

        X, y = generate_data(
            scenario,
            RANDOM_SEED + trial,
            true_cnts_idx,
            true_bin_idx,
        )
        X_jitter = jitter_binary_features(
            X,
            seed=RANDOM_SEED + 1_000_000 + trial,
        )
        print(f"Trial {trial + 1}/{N_TRIALS}", flush=True)

        for method_index, (method_name, config) in enumerate(METHODS.items()):
            np.random.seed(
                RANDOM_SEED + 100_000 * trial + 1_000 * method_index
            )
            ipss_args = {
                "selector": config["selector"],
                "n_jobs": -1,
            }
            if config["delta"] is not None:
                ipss_args["delta"] = config["delta"]

            X_input = X_jitter if config["jitter"] else X
            result = ipss(X_input, y, **ipss_args)
            q_values = result["q_values"]
            selected_per_alpha = {
                alpha_index: {
                    feature
                    for feature, q_value in q_values.items()
                    if q_value <= alpha
                }
                for alpha_index, alpha in enumerate(ALPHAS)
            }
            record_selection(
                metrics,
                selected_per_alpha,
                trial,
                method_index,
                true_cont,
                true_bin,
            )
            print(f"  Finished {method_name}", flush=True)

    if save_data or save_fig:
        output_dir = resolve_output_dir(OUTPUT_ROOT / f"{scenario}_{SIGNAL_TYPE}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    if save_data:
        save_results(metrics, true_features_by_trial, output_dir)
    save_plots(metrics, output_dir)

    if output_dir is not None:
        print(f"Results written to {output_dir.resolve()}", flush=True)
    else:
        print("Test run: nothing written to disk", flush=True)


def main():
    for scenario in SCENARIOS:
        run_scenario(scenario)


if __name__ == "__main__":
    main()
