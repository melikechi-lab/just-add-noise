# IPSS selection counts for the ML datasets
"""Run IPSS at a target FDR with each selector on each ML dataset and record how
many continuous vs categorical predictors are selected.

Single seed: IPSS already averages over B subsamples, so its selected set is far
more stable than a single importance fit.

The selector configuration mirrors Jiahe_simulations/main_simulations.py: a
one-time uniform jitter is added to the categorical columns for the jittered
selectors, and `delta` (the IPSS probability-measure exponent) is set to 2 for
the jittered and UFI selectors, left at the IPSS default otherwise.

Outputs (ml_datasets/):
  ipss_selection_counts.csv    one row per dataset x selector
  ipss_selection_details.json  selected predictor names + IPSS q-values,
                               per dataset x selector
Then prints a plain-text summary and a LaTeX table.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from clean_selected_datasets import load_and_clean_dataset, preprocess_full_dataset
from extra_datasets import QSAR_BIODEG_DESCRIPTORS, load_extra_dataset

try:
    from ipss import ipss
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f'this script requires the ipss package ({exc})')

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
output_dir = Path(__file__).resolve().parent / 'ml_datasets'
counts_path = output_dir / 'ipss_selection_counts.csv'
details_path = output_dir / 'ipss_selection_details.json'

save_results = True
target_fdr = 0.1   # nominal FDR level in Figure S2
random_seed = 302
jitter_strength = 1e-4
n_jobs = -1

# selectors to run.  key -> dict with:
#   selector : ipss base selector ('rf', 'gb', 'ufi')
#   jitter   : add the one-time categorical jitter before fitting
#   delta    : IPSS probability-measure exponent; None uses the IPSS default
#              (1.25 for rf/ufi, 1 for gb).  main_simulations.py uses 2 for the
#              jittered and UFI selectors.
methods = {
    'RF':        {'selector': 'rf',  'jitter': False, 'delta': None, 'B': 200},
    'RF-jitter': {'selector': 'rf',  'jitter': True,  'delta': None, 'B': 200},
    'GB':        {'selector': 'gb',  'jitter': False, 'delta': None, 'B': 200},
    'GB-jitter': {'selector': 'gb',  'jitter': True,  'delta': None, 'B': 200},
    'UFI':       {'selector': 'ufi', 'jitter': False, 'delta': None, 'B': 200},
}

# Hepatitis (n=155, almost every predictor prognostic) is left out of this table;
# its saved output files under ml_datasets/ remain in place.
dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    'forest_fires': 'Forest Fires',
    'student_performance': 'Student Performance',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'german_credit': 'German Credit',
    'qsar_biodeg': 'QSAR Biodegradation',
    'hypothyroid': 'Hypothyroid',
}

# OpenML/PMLB datasets loaded with the generic recipe in
# extra_datasets.load_extra_dataset
extra_specs = {
    'german_credit': {'source': 'openml', 'id': 31, 'target': None,
                      'task': 'classification', 'title': 'German Credit'},
    'qsar_biodeg': {'source': 'openml', 'id': 1494, 'target': None,
                    'task': 'classification', 'title': 'QSAR Biodegradation',
                    'rename': QSAR_BIODEG_DESCRIPTORS},
    'hypothyroid': {'source': 'pmlb', 'id': 'hypothyroid', 'target': None,
                    'task': 'classification', 'title': 'Hypothyroid'},
}

# datasets to (re)compute this run; when merge_with_saved, the rest of
# dataset_titles is kept from the saved counts CSV / details JSON
# compute every dataset from scratch; set merge_with_saved = True and a shorter
# `datasets` list to update only some rows against an existing run.
datasets = list(dataset_titles)
merge_with_saved = False
titles = dataset_titles


def load_dataset(key):
    """Return (X_df, y_series, cat_idx, task, title)."""
    if key in extra_specs:
        X_df, y_series, cat_idx, task, *_ = load_extra_dataset(extra_specs[key])
    else:
        dataset = load_and_clean_dataset(key)
        X_df, y_series, cat_idx = preprocess_full_dataset(dataset)
        task = dataset.task
    return X_df, y_series, cat_idx, task, dataset_titles[key]


#----------------------------------------------------------------
# Run IPSS for every dataset x selector
#----------------------------------------------------------------
def jitter_categoricals(X, cat_idx):
    """One fixed uniform jitter realization on the categorical columns."""
    rng = np.random.default_rng(random_seed)
    out = X.astype(float, copy=True)
    out[:, cat_idx] += rng.uniform(
        -jitter_strength, jitter_strength, size=(X.shape[0], len(cat_idx))
    )
    return out


count_rows = []
details = {}

for key in datasets:
    X_df, y_series, cat_idx, task, title = load_dataset(key)
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy()
    feature_names = list(X_df.columns)
    categorical = set(cat_idx)
    n, p = X.shape
    n_categorical = len(cat_idx)

    print(f'{title}: n={n}, p={p}, {n_categorical} categorical, task={task}')
    X_jittered = jitter_categoricals(X, cat_idx)

    for name, config in methods.items():
        np.random.seed(random_seed)
        X_input = X_jittered if config['jitter'] else X
        kwargs = {'selector': config['selector'], 'target_fdr': target_fdr, 'n_jobs': n_jobs, 'B': config['B']}
        if config['delta'] is not None:
            kwargs['delta'] = config['delta']

        result = ipss(X_input, y, **kwargs)
        q_values = {int(i): float(q) for i, q in result['q_values'].items()}
        by_q = sorted(q_values, key=lambda i: q_values[i])
        selected = set(int(i) for i in result['selected_features'])

        def as_entries(indices):
            return [{'name': feature_names[i], 'q_value': q_values[i]} for i in indices]

        selected_categorical = as_entries(i for i in by_q if i in selected and i in categorical)
        selected_continuous = as_entries(i for i in by_q if i in selected and i not in categorical)

        count_rows.append({
            'dataset': key,
            'selector': name,
            'n_predictors': p,
            'n_categorical': n_categorical,
            'n_selected': len(selected),
            'categorical_selected': len(selected_categorical),
            'continuous_selected': len(selected_continuous),
        })
        details.setdefault(key, {})[name] = {
            'categorical': selected_categorical,
            'continuous': selected_continuous,
        }
        print(
            f'  {name:10s}  selected {len(selected):2d}  '
            f'({len(selected_categorical)}/{n_categorical} categorical, '
            f'{len(selected_continuous)} continuous)'
        )
    print()

counts = pd.DataFrame(count_rows)

if merge_with_saved:
    if counts_path.exists():
        prior = pd.read_csv(counts_path)
        prior = prior[prior['dataset'].isin(dataset_titles) & ~prior['dataset'].isin(datasets)]
        counts = pd.concat([prior, counts], ignore_index=True)
    if details_path.exists():
        with open(details_path) as handle:
            saved_details = json.load(handle)
        for key, entry in saved_details.items():
            if key in dataset_titles and key not in datasets:
                details.setdefault(key, entry)

# datasets to show in the printouts: everything now present, in dataset_titles order
report_datasets = [k for k in dataset_titles if k in set(counts['dataset'])]

if save_results:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, index=False)
    with open(details_path, 'w') as handle:
        json.dump(details, handle, indent=2)
    print(f'saved {counts_path}')
    print(f'saved {details_path}')

#----------------------------------------------------------------
# Plain-text summary: categoricals selected (of total), per dataset x selector
#----------------------------------------------------------------
selector_names = list(methods)
print()
print(f'Categorical predictors selected at target FDR {target_fdr}  '
      f'(categorical selected / total selected)')
print()
width = 12
name_width = 28
print(f'{"dataset":{name_width}s}{"C":>4s}' + ''.join(f'{s:>{width}s}' for s in selector_names))
print('-' * (name_width + 4 + width * len(selector_names)))
for key in report_datasets:
    sub = counts[counts['dataset'] == key].set_index('selector')
    C = int(sub['n_categorical'].iloc[0])
    line = f'{titles[key]:{name_width}s}{C:>4d}'
    for s in selector_names:
        row = sub.loc[s]
        line += f'{f"{row.categorical_selected}/{row.n_selected}":>{width}s}'
    print(line)

#----------------------------------------------------------------
# Selected predictor names, per dataset x selector
#----------------------------------------------------------------
print()
print('=' * 64)
print('Selected predictors  (categorical, then continuous), with IPSS q-values')


def _format_entries(entries):
    return ', '.join(f'{e["name"]} (q={e["q_value"]:.3f})' for e in entries) or '(none)'


for key in report_datasets:
    print()
    print(titles[key])
    for name in selector_names:
        selected = details[key][name]
        print(f'  {name}')
        print(f'    cat:  {_format_entries(selected["categorical"])}')
        print(f'    cont: {_format_entries(selected["continuous"])}')

#----------------------------------------------------------------
# LaTeX table
#----------------------------------------------------------------
print()
print('=' * 64)
print()
print(r'\begin{table}[htbp]')
print(r'\centering')
print(r'\small')
print(r'\setlength{\tabcolsep}{6pt}')
print(rf'\begin{{tabular}}{{lc{"c" * len(selector_names)}}}')
print(r'\toprule')
print(' & '.join(['Dataset', '$C$'] + selector_names) + r' \\')
print(r'\midrule')
for key in report_datasets:
    sub = counts[counts['dataset'] == key].set_index('selector')
    C = int(sub['n_categorical'].iloc[0])
    cells = [titles[key], str(C)]
    for s in selector_names:
        row = sub.loc[s]
        cells.append(f'{row.categorical_selected}/{row.n_selected}')
    print(' & '.join(cells) + r' \\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(
    r'\caption{\textit{Categorical predictors selected by IPSS}. For each '
    rf'selector at target FDR ${target_fdr}$, the number of categorical '
    r'predictors selected over the total number of predictors selected; $C$ is '
    r'the number of categorical predictors available. Single seed. Jittered '
    r'selectors add a one-time uniform jitter to the categorical columns.}'
)
print(r'\label{tab:ipss_selection}')
print(r'\end{table}')
