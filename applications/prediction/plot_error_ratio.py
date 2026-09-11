# Prediction error-ratio figure (Figure S5)
"""Box-and-whisker of the jittered/unjittered test-error ratio across PMLB datasets,
one box per model family. Reads the per-method CSVs written by run_prediction.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
dataset_family = 'pmlb'
input_prefix = str(script_dir / 'results' / f'error_analysis_{dataset_family}')
metadata_csv = script_dir / 'dataset_lists' / 'clean_datasets_pmlb_final.csv'

# jitter methods' result CSVs are tagged with the jitter strength they were run at
# (see run_prediction.py); pick which sweep to load here
jitter_strength = 1e-4

def is_jitter_method(method):
    return method.endswith('_one_time') or method.endswith('_per_tree')

def method_csv_path(method):
    tag = f'_j{jitter_strength:.0e}' if is_jitter_method(method) else ''
    return f'{input_prefix}_{method}{tag}.csv'

# (numerator_method, denominator_method) pairs to plot as error ratios
ratio_pairs_to_plot = [
    ('rf_one_time', 'rf'),
    # ('rf_one_time', 'catboost'),
    # ('rf_one_time', 'tabpfn'),
    # ('tabpfn_one_time', 'tabpfn'),
    # ('xgb_one_time', 'catboost'),
    ('xgb_one_time', 'xgb'),
]

# clean display names for the paper box-and-whisker figure (keyed by numerator method)
method_display_names = {
    'rf_one_time': 'RF',
    'xgb_one_time': 'XGBoost',
}

# dataset characteristics to plot error ratios against (any column produced below)
x_params = ['n_rows', 'n_predictors', 'cat_to_cont_ratio']

# None for classification + regression; else 'classification' or 'regression'
task_filter = None

plot_scatter = False
plot_histograms = False
plot_boxplots = True

# True plots log(ratio) (symmetric, better for aggregating/comparing many datasets);
# False plots the raw ratio directly (numerator error as a fraction of denominator error)
use_log_ratio = False

n_top_outliers = None

fit_linear_model = False  # set True for the log-ratio ~ dataset-size OLS diagnostic (needs statsmodels)
regression_covariates = ['n_rows', 'n_continuous', 'n_categorical']  # not n_predictors: n_predictors = n_continuous + n_categorical exactly, so including all three is collinear

# same predictor matrix, different targets (confirmed by inspecting the downloaded data) --
# excluded from the regression so this one family of 6 doesn't dominate the fit; not excluded
# elsewhere (plots, outlier tables) since those aren't sensitive to non-independence the same way
thyroid_family_duplicates = ['allhypo', 'allhyper', 'allbp', 'allrep', 'dis']  # keep 'hypothyroid'

# print name/n_rows/n_predictors for datasets whose (raw, non-log) ratio exceeds this; None to skip
ratio_cutoff = 1.5

# exclude a dataset from ratio pai analysis if both methods' errors are below this
min_error_threshold = 0.05

save_fig = True

#----------------------------------------------------------------
# Join results (by name) to dataset metadata, derive plotting variables
#----------------------------------------------------------------
# result CSVs have a 'did' column for OpenML-sourced runs, but not for PMLB-sourced runs
# (see prediction.py); 'name' is present either way, so it's the merge key
# used throughout here, both for joining the per-method result CSVs to each other and
# for joining the combined results to metadata_csv
methods = sorted(set(m for pair in ratio_pairs_to_plot for m in pair))

results = None
join_cols = None
for method in methods:
    method_df = pd.read_csv(method_csv_path(method))
    method_df = method_df.drop(columns=[c for c in ['jitter_strength'] if c in method_df.columns])
    if results is None:
        results = method_df
        join_cols = [c for c in ['did', 'name', 'task'] if c in method_df.columns]
    else:
        results = results.merge(method_df, on=join_cols, how='outer')

metadata = pd.read_csv(metadata_csv)[['name', 'n_rows', 'n_predictors', 'n_continuous', 'n_categorical']]

data = results.merge(metadata, on='name', how='left')
data['cat_to_cont_ratio'] = np.where(data['n_continuous'] > 0, data['n_categorical'] / data['n_continuous'], np.nan)

id_cols = [c for c in ['did', 'name'] if c in data.columns]

if task_filter is not None:
    data = data[data['task'] == task_filter]

# results only save raw errors; derive numerator_error / denominator_error here.
# ratios is a list of dicts carrying both method names alongside the derived column
# names, so downstream code never has to re-parse a method name out of a string
ratios = []
for numerator, denominator in ratio_pairs_to_plot:
    error_num_col, error_den_col = f'error_{numerator}', f'error_{denominator}'

    stable = (data[error_den_col] >= min_error_threshold) | (data[error_num_col] >= min_error_threshold)

    ratio_col = f'ratio_{numerator}_over_{denominator}'
    data[ratio_col] = data[error_num_col] / data[error_den_col].where(data[error_den_col] > 0)
    data.loc[~stable, ratio_col] = np.nan

    log_col = f'log_ratio_{numerator}_over_{denominator}'
    data[log_col] = np.log(data[ratio_col].where(data[ratio_col] > 0))

    ratios.append({
        'numerator': numerator,
        'denominator': denominator,
        'label': f'{numerator} / {denominator}',
        'error_num_col': error_num_col,
        'error_den_col': error_den_col,
        'ratio_col': ratio_col,
        'log_col': log_col,
        'stable': stable,
        'plot_col': log_col if use_log_ratio else ratio_col,
        'ref_line': 0 if use_log_ratio else 1,
        'axis_label': 'log(numerator error / denominator error)' if use_log_ratio else 'numerator error / denominator error',
    })

print(f"{len(data)} datasets, plotting {[r['label'] for r in ratios]} vs {x_params}")

#----------------------------------------------------------------
# Datasets excluded by min_error_threshold
#----------------------------------------------------------------
for r in ratios:
    has_both_errors = data[r['error_num_col']].notna() & data[r['error_den_col']].notna()
    excluded = data.loc[has_both_errors & ~r['stable'], [*id_cols, 'n_rows', 'n_predictors']].drop_duplicates()
    print(f"\n{len(excluded)} datasets excluded from {r['label']} (both errors < {min_error_threshold}):")
    print(excluded.to_string(index=False))

#----------------------------------------------------------------
# Datasets above a ratio cutoff
#----------------------------------------------------------------
if ratio_cutoff is not None:
    for r in ratios:
        above = data.loc[r['stable'] & (data[r['ratio_col']] > ratio_cutoff), [*id_cols, 'n_rows', 'n_predictors', r['ratio_col']]].dropna().sort_values(r['ratio_col'], ascending=False)
        print(f"\n{len(above)} datasets with {r['label']} ratio > {ratio_cutoff}:")
        print(above.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

#----------------------------------------------------------------
# Top outliers on either side
#----------------------------------------------------------------
if n_top_outliers is not None and n_top_outliers > 0:
    for r in ratios:
        ranked = data.loc[r['stable'], [*id_cols, r['error_den_col'], r['error_num_col'], r['ratio_col']]].dropna().sort_values(r['ratio_col'], ascending=False)

        top_den = ranked.head(n_top_outliers)
        print(f"\ntop {len(top_den)} datasets where {r['denominator']} most outperforms {r['numerator']}:")
        print(top_den.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

        top_num = ranked.tail(n_top_outliers).iloc[::-1]
        print(f"\ntop {len(top_num)} datasets where {r['numerator']} most outperforms {r['denominator']}:")
        print(top_num.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

#----------------------------------------------------------------
# Linear model: log ratio ~ dataset characteristics
#----------------------------------------------------------------
if fit_linear_model:
    import statsmodels.api as sm  # optional; only needed for this diagnostic
    for r in ratios:
        reg_data = data.loc[r['stable'] & ~data['name'].isin(thyroid_family_duplicates), ['name', r['log_col'], *regression_covariates]].dropna()

        if len(reg_data) < len(regression_covariates) + 2:
            print(f"\nnot enough datasets to fit a linear model for {r['label']} ({len(reg_data)} available)")
            continue

        X = sm.add_constant(reg_data[regression_covariates])
        y = reg_data[r['log_col']]
        model = sm.OLS(y, X).fit()

        print(f"\nlinear model: log({r['label']}) ~ {' + '.join(regression_covariates)}  (n={len(reg_data)})")
        print(model.summary())

#----------------------------------------------------------------
# Plot log error ratio vs each dataset characteristic
#----------------------------------------------------------------
if plot_scatter and len(ratios) == 0:
    print('no ratio pairs to plot')
elif plot_scatter:
    fig, axes = plt.subplots(len(ratios), len(x_params), figsize=(16,6))
    axes = np.atleast_1d(axes).flatten().reshape(len(ratios), len(x_params))
    for i, x_param in enumerate(x_params):
        for j, r in enumerate(ratios):
            ax = axes[j][i]
            plot_data = data[[x_param, r['plot_col']]].dropna()
            ax.scatter(plot_data[x_param], plot_data[r['plot_col']])
            ax.axhline(r['ref_line'], color='red', linestyle='--', label=f"ratio = {r['ref_line']}")
            ax.set_xlabel(x_param)
            ax.set_ylabel(r['axis_label'])
            ax.set_title(r['label'])
            ax.legend()

    plt.tight_layout()
    if save_fig:
        fig.savefig('jitter_error_ratio_vs_params.png')
    plt.show()

#----------------------------------------------------------------
# Plot histograms of log error ratios
#----------------------------------------------------------------
if plot_histograms and len(ratios) == 0:
    print('no ratio pairs to plot')
elif plot_histograms:
    fig, axes = plt.subplots(1, len(ratios), figsize=(16,6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ratios):
        ratio_values = data[r['plot_col']].dropna()
        ax.hist(ratio_values, bins=30, edgecolor='black')
        ax.axvline(r['ref_line'], color='red', linestyle='--', label=f"ratio = {r['ref_line']}")
        ax.set_xlabel(r['axis_label'])
        ax.set_ylabel('count')
        ax.set_title(r['label'])
        ax.legend()

    plt.tight_layout()
    if save_fig:
        fig.savefig('jitter_log_error_ratio_hist.png')
    plt.show()

#----------------------------------------------------------------
# Box and whisker of log error ratios
#----------------------------------------------------------------
if plot_boxplots and len(ratios) == 0:
    print('no ratio pairs to plot')
elif plot_boxplots:
    # paper figure: one horizontal box per model family, first ratio on top
    positions = list(range(len(ratios), 0, -1))
    box_data = [data[r['plot_col']].dropna() for r in ratios]
    labels = [method_display_names.get(r['numerator'], r['label']) for r in ratios]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.boxplot(
        box_data,
        positions=positions,
        tick_labels=labels,
        widths=0.55,
        vert=False,
        showfliers=False,
        medianprops=dict(linewidth=2.5),
    )
    ax.axvline(ratios[0]['ref_line'], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Error ratio', fontsize=16)
    ax.set_ylim(0.4, len(ratios) + 0.6)

    plt.tight_layout()
    if save_fig:
        fig.savefig('error_ratio_boxplots.png', dpi=300)
    plt.show()
