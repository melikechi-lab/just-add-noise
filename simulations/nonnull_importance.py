"""
Generate the non-null MDI convergence panel for

    Y = X + Z + epsilon,

where X ~ N(0,1), Z is equally likely to be -1 or 1,
and epsilon ~ N(0,1).

The script compares:
    1. Standard fully grown CART MDI.
    2. One-time jittered CART MDI.

Outputs:
    figure_nonnull_importance.pdf
    figure_nonnull_importance.png
    figure_nonnull_importance.csv
"""

from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

OUTPUT_DIR = Path(__file__).resolve().parent

#----------------------------------------------------------------
# Simulation settings
#----------------------------------------------------------------
SAMPLE_SIZES = np.array([25, 50, 100, 200, 400, 800, 1600])
N_REPETITIONS = 200
MASTER_SEED = 123
JITTER_STRENGTH = 0.0001

#----------------------------------------------------------------
# Figure settings
#----------------------------------------------------------------
save_figure = True
output_stem = "figure_nonnull_importance"

fig_width = 10
fig_height = 6
savefig_dpi = 300

line_width = 2
line_alpha = 1
marker_size = 10
axis_label_fontsize = 18
tick_label_fontsize = 12
legend_fontsize = 15

reference_lines = [1/3, 1/2, 2/3]
y_ticks = [0, 1/3, 1/2, 2/3, 1]
y_tick_labels = [r'$0$', r'$1/3$', r'$1/2$', r'$2/3$', r'$1$']

x_label = r"Sample size $(n)$"
y_label = "Normalized MDI"

# curves to plot: (method, feature) -> legend label
series = {
    ('Standard CART', 'X'): r'$X$ (no noise)',
    ('Standard CART', 'Z'): r'$Z$ (no noise)',
    ('Jittered CART', 'X'): r'$X$ (with noise)',
    ('Jittered CART', 'Z'): r'$Z$ (with noise)',
}

# color encodes the feature, linestyle encodes the method
feature_colors = {
    'X': 'C0',
    'Z': 'C1',
}
method_styles = {
    'Standard CART': {'linestyle': '-', 'marker': 'o'},
    'Jittered CART': {'linestyle': '--', 'marker': 'o'},
}


#----------------------------------------------------------------
# Simulation
#----------------------------------------------------------------
def generate_data(n, rng):
    x = rng.normal(0.0, 1.0, size=n)
    z = rng.choice([-1.0, 1.0], size=n)
    epsilon = rng.normal(0.0, 1.0, size=n)
    y = x + z + epsilon
    predictors = np.column_stack([x, z])
    return predictors, y


def fit_fully_grown_tree_importance(predictors, y, random_state):
    model = DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        ccp_alpha=0.0,
        random_state=random_state,
    )
    model.fit(predictors, y)

    importance = np.asarray(model.feature_importances_, dtype=float)
    total = importance.sum()
    if total > 0:
        importance = importance / total
    return importance


def jitter_binary_predictor(predictors, rng, strength):
    jittered = predictors.copy()
    jittered[:, 1] += rng.uniform(
        low=-strength,
        high=strength,
        size=jittered.shape[0],
    )
    return jittered


def run_simulation():
    master_rng = np.random.default_rng(MASTER_SEED)
    rows = []

    for n in SAMPLE_SIZES:
        standard_importance = np.zeros((N_REPETITIONS, 2))
        jittered_importance = np.zeros((N_REPETITIONS, 2))

        for repetition in range(N_REPETITIONS):
            data_seed = int(master_rng.integers(0, 2**32 - 1))
            tree_seed = int(master_rng.integers(0, 2**32 - 1))
            jitter_seed = int(master_rng.integers(0, 2**32 - 1))

            data_rng = np.random.default_rng(data_seed)
            predictors, y = generate_data(n, data_rng)

            standard_importance[repetition] = fit_fully_grown_tree_importance(
                predictors,
                y,
                random_state=tree_seed,
            )

            jitter_rng = np.random.default_rng(jitter_seed)
            jittered_predictors = jitter_binary_predictor(
                predictors,
                rng=jitter_rng,
                strength=JITTER_STRENGTH,
            )
            jittered_importance[repetition] = fit_fully_grown_tree_importance(
                jittered_predictors,
                y,
                random_state=tree_seed,
            )

        for method_name, values in [
            ("Standard CART", standard_importance),
            ("Jittered CART", jittered_importance),
        ]:
            for feature_index, feature_name in enumerate(["X", "Z"]):
                mean_value = values[:, feature_index].mean()
                standard_error = (
                    values[:, feature_index].std(ddof=1)
                    / np.sqrt(N_REPETITIONS)
                )
                rows.append(
                    {
                        "n": n,
                        "method": method_name,
                        "feature": feature_name,
                        "mean_importance": mean_value,
                        "standard_error": standard_error,
                    }
                )

    return pd.DataFrame(rows)


#----------------------------------------------------------------
# Figure
#----------------------------------------------------------------
def make_figure(results):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for y_value in reference_lines:
        ax.axhline(y_value, color='gray', linestyle='--', linewidth=1.5)

    legend_handles = []
    for (method, feature), label in series.items():
        subset = results[
            (results["method"] == method)
            & (results["feature"] == feature)
        ].sort_values("n")

        color = feature_colors[feature]
        style = method_styles[method]

        ax.errorbar(
            subset['n'],
            subset['mean_importance'],
            yerr=1.96 * subset['standard_error'],
            color=color,
            alpha=line_alpha,
            linestyle=style['linestyle'],
            linewidth=line_width,
            marker=style['marker'],
            markersize=marker_size,
            capsize=3,
        )
        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle=style['linestyle'],
                linewidth=line_width,
                label=label,
            )
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(n) for n in SAMPLE_SIZES])
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.legend(handles=legend_handles, fontsize=legend_fontsize)

    plt.tight_layout()

    if save_figure:
        fig.savefig(OUTPUT_DIR / f"{output_stem}.png", dpi=savefig_dpi)
        fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf")

    plt.show()

if __name__ == "__main__":
    simulation_results = run_simulation()
    simulation_results.to_csv(
        OUTPUT_DIR / f"{output_stem}.csv",
        index=False,
    )
    make_figure(simulation_results)
