# Random forest / XGBoost vs their one-time-jittered counterparts across PMLB datasets
"""Figure S5 (fig:prediction_error_pmlb): fit RF and XGBoost with and without a
one-time jitter of strength delta = 1e-4, 5-fold CV, and record the jittered/unjittered
test-error ratio for every dataset in dataset_lists/clean_datasets_pmlb_final.csv.
Writes one CSV per method to results/; plot with plot_error_ratio.py.
"""

import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import pmlb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from methods.dataset_types import classify_column
from predict_methods import jitterRF, jitterXGB

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_results = True
save_fig = True
max_datasets_to_run = None

dataset_family = 'pmlb'

script_dir = Path(__file__).resolve().parent
input_csv = script_dir / 'dataset_lists' / 'clean_datasets_pmlb_final.csv'
output_prefix = str(script_dir / 'results' / 'error_analysis')
pmlb_cache_dir = str(script_dir / 'pmlb_cache')  # local disk cache so re-runs don't re-hit the network
os.makedirs(script_dir / 'results', exist_ok=True)

fetch_retries = 3
fetch_retry_delay = 5

n_folds = 5
random_seed = 302

n_estimators = 100
max_depth = None
rf_args = {'n_estimators': n_estimators, 'max_depth': max_depth}
xgb_args = {'n_estimators': n_estimators, 'max_depth': max_depth}
jitter_strength = 0.0001  # delta in the paper

methods_to_run = ['rf', 'rf_one_time', 'xgb', 'xgb_one_time']

#----------------------------------------------------------------
# Load candidate datasets, deduplicating near-identical entries
#----------------------------------------------------------------
candidates = pd.read_csv(input_csv)

# an OpenML-sourced input_csv (has a 'did' column) can contain the same underlying
# data re-uploaded under a different did/target (e.g. 17 different 'anneal' entries),
# resampled under different seeds (e.g. 'covertype_seed_0_nrows_2000_...' through
# '..._seed_4_...'), near-identical generated variants (e.g. 9
# 'jungle_chess_2pcs_endgame_<piece>_<piece>' entries), or a '_clean' re-upload of a raw
# dataset also present (e.g. 'autos' and 'autos_clean') -- keep one per base name,
# preferring the '_clean' version when both exist, so these near-duplicates don't
# dominate the results below; PMLB-sourced input_csv (no 'did' column) has no such
# duplicates, so this is a no-op there
has_did = 'did' in candidates.columns
sort_key = 'did' if has_did else 'name'

candidates['dedup_name'] = candidates['name'].str.replace(r'_seed_\d+', '', regex=True)
candidates['dedup_name'] = candidates['dedup_name'].str.replace(r'jungle_chess_2pcs_endgame_\w+_\w+', 'jungle_chess_2pcs_endgame', regex=True)
candidates['is_clean'] = candidates['dedup_name'].str.endswith('_clean')
candidates['dedup_name'] = candidates['dedup_name'].str.replace(r'_clean$', '', regex=True)
candidates = candidates.sort_values(['dedup_name', 'is_clean', sort_key], ascending=[True, False, True])
candidates = candidates.drop_duplicates(subset='dedup_name', keep='first').drop(columns=['dedup_name', 'is_clean']).sort_values(sort_key)

if max_datasets_to_run is not None:
    candidates = candidates.head(max_datasets_to_run)

print(f'{len(candidates)} datasets after deduplication')

#----------------------------------------------------------------
# Methods
#----------------------------------------------------------------
method_functions = {
    'rf': lambda X_train, X_test, y_train, y_test, task, cat_idx: jitterRF(X_train, X_test, y_train, y_test, task=task, jitter_method=None, **rf_args),
    'rf_one_time': lambda X_train, X_test, y_train, y_test, task, cat_idx: jitterRF(X_train, X_test, y_train, y_test, task=task, jitter_method='one_time', jitter_strength=jitter_strength, cat_idx=cat_idx, **rf_args),
    'rf_per_tree': lambda X_train, X_test, y_train, y_test, task, cat_idx: jitterRF(X_train, X_test, y_train, y_test, task=task, jitter_method='per_tree', jitter_strength=jitter_strength, cat_idx=cat_idx, **rf_args),
    'xgb': lambda X_train, X_test, y_train, y_test, task, cat_idx: jitterXGB(X_train, X_test, y_train, y_test, task=task, jitter_method=None, **xgb_args),
    'xgb_one_time': lambda X_train, X_test, y_train, y_test, task, cat_idx: jitterXGB(X_train, X_test, y_train, y_test, task=task, jitter_method='one_time', jitter_strength=jitter_strength, cat_idx=cat_idx, **xgb_args),
}

# jitter_method -> base_method pairs to compare; only kept when both sides are in methods_to_run
all_ratio_pairs = {'rf_one_time': 'rf', 'rf_per_tree': 'rf', 'xgb_one_time': 'xgb'}
ratio_pairs = {j: b for j, b in all_ratio_pairs.items() if j in methods_to_run and b in methods_to_run}

id_cols = ['did', 'name'] if has_did else ['name']

def fetch_with_retry(fetch_fn):
    # both openml and pmlb downloads occasionally hit transient connection drops
    # (e.g. RemoteDisconnected from the host closing the connection mid-request);
    # retry a few times with a short delay before giving up on the dataset
    for attempt in range(1, fetch_retries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt == fetch_retries:
                raise
            print(f'    fetch failed ({e}), retrying ({attempt}/{fetch_retries})...')
            time.sleep(fetch_retry_delay)

results = []
for i, row in enumerate(candidates.itertuples(), start=1):
    label = f'did={row.did} ({row.name})' if has_did else row.name
    print(f'[{i}/{len(candidates)}] {label}...')

    try:
        if has_did:
            def _fetch():
                d = openml.datasets.get_dataset(row.did, download_data=True)
                return d.get_data(target=d.default_target_attribute)
            X_df, y_raw, _, feature_names = fetch_with_retry(_fetch)
        else:
            # pmlb pre-encodes every column (including the target) to numeric, so there's
            # no target dtype to read the task off of -- use pmlb's own task-type lists instead
            df = fetch_with_retry(lambda: pmlb.fetch_data(row.name, local_cache_dir=pmlb_cache_dir))
            feature_names = [c for c in df.columns if c != 'target']
            X_df = df[feature_names]
            y_raw = df['target']

        # classify predictors from the actual downloaded values (see dataset_types.classify_column)
        # rather than trusting OpenML's declared numeric/nominal type; constant columns carry no
        # information under either label and are dropped
        kinds = {name: classify_column(X_df[name])[0] for name in feature_names}
        keep_features = [name for name in feature_names if kinds[name] != 'constant']
        X_df = X_df[keep_features]

        n, p = X_df.shape
        X = np.zeros((n, p))
        for j, name in enumerate(keep_features):
            col = X_df[name]
            if kinds[name] == 'categorical':
                X[:, j] = LabelEncoder().fit_transform(col.astype(str))
            else:
                X[:, j] = col.astype(float)

        if has_did:
            is_classification = y_raw.dtype.name in ('category', 'object')
        else:
            # pmlb's own bundled classification_dataset_names/regression_dataset_names lists are
            # version-pinned and occasionally wrong (e.g. titanic's actual metadata.yaml on GitHub
            # says task: classification, but the installed pmlb package lists it under regression)
            # -- a binary target is unambiguous, so use that as a safety override
            is_classification = row.name in pmlb.classification_dataset_names or y_raw.nunique() == 2

        if is_classification:
            task = 'classification'
            y = LabelEncoder().fit_transform(y_raw.astype(str))
        else:
            task = 'regression'
            y = y_raw.to_numpy(dtype=float)

        cat_idx = [j for j, name in enumerate(keep_features) if kinds[name] == 'categorical']

        if task == 'classification':
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

        errors = {method: [] for method in methods_to_run}
        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for method in methods_to_run:
                error = method_functions[method](X_train, X_test, y_train, y_test, task, cat_idx)['error']
                errors[method].append(error)

        mean_errors = {method: np.mean(errors[method]) for method in methods_to_run}

        result = {col: getattr(row, col) for col in id_cols}
        result['task'] = task
        for method in methods_to_run:
            result[f'error_{method}'] = mean_errors[method]
        results.append(result)

    except Exception as e:
        print(f'  skipping {label}: {e}')
        continue

#----------------------------------------------------------------
# Write output
#----------------------------------------------------------------
# jitter methods use jitter_strength, so their output is tagged with it (e.g. '_j1e-04')
# to keep sensitivity-sweep runs from overwriting each other; non-jitter base methods are
# strength-independent and keep the plain name
def is_jitter_method(method):
    return method.endswith('_one_time') or method.endswith('_per_tree')

def method_csv_path(method):
    tag = f'_j{jitter_strength:.0e}' if is_jitter_method(method) else ''
    return f'{output_prefix}_{dataset_family}_{method}{tag}.csv'

results_df = pd.DataFrame(results)
if save_results:
    for method in methods_to_run:
        method_csv = method_csv_path(method)
        method_df = results_df[[*id_cols, 'task', f'error_{method}']].copy()
        if is_jitter_method(method):
            method_df['jitter_strength'] = jitter_strength
        method_df.to_csv(method_csv, index=False)
    print(f'\nran on {len(results_df)}/{len(candidates)} datasets, written {[method_csv_path(m) for m in methods_to_run]}')
else:
    print(f'\nran on {len(results_df)}/{len(candidates)} datasets (save_results = False, not written)')

#----------------------------------------------------------------
# Plot histograms of jitter_error / base_error ratios
#----------------------------------------------------------------
if len(ratio_pairs) == 0:
    print('no ratio pairs to plot (methods_to_run is missing one side of every pair in all_ratio_pairs)')
else:
    fig, axes = plt.subplots(1, len(ratio_pairs), figsize=(6 * len(ratio_pairs), 5))
    axes = np.atleast_1d(axes)
    for ax, (jitter_method, base_method) in zip(axes, ratio_pairs.items()):
        base_error = results_df[f'error_{base_method}']
        ratios = (results_df[f'error_{jitter_method}'] / base_error.where(base_error > 0)).dropna()
        ax.hist(ratios, bins=30, edgecolor='black')
        ax.axvline(1, color='red', linestyle='--', label='ratio = 1')
        ax.set_xlabel('jitter error / base error')
        ax.set_ylabel('count')
        ax.set_title(f'{jitter_method} vs {base_method}')
        ax.legend()

    plt.tight_layout()
    if save_fig:
        fig.savefig('jitter_error_ratios.png')
    plt.show()
