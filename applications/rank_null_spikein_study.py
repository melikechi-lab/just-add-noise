# Null spike-in ranking study for the ML datasets
"""Add synthetic null predictors to each ML dataset and record where they land in
the importance ranking, under standard RF, one-time and per-tree jittered RF,
and UFI.

The nulls are permuted copies of real columns (independent row permutation:
same marginal, no association with y), so a null continuous predictor and a null
categorical predictor are equivalent in everything but type. A fair importance
measure ranks them the same on average; standard MDI ranks the continuous null
higher. The gap between the two, and how jittering closes it, is the real-data
analogue of the null-model theory.

Outputs (ml_datasets/), when save_results:
  rank_null_spikein_by_position.csv  dataset x method x seed x rank -> kind
                                     (real / null_continuous / null_categorical);
                                     feeds plot_rank_null_spikein_heatmap.py
  rank_null_spikein_summary.csv      dataset x method: mean percentile of the two
                                     null types and their difference
Then prints a plain-text summary and a LaTeX table.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from clean_selected_datasets import load_and_clean_dataset, preprocess_full_dataset
from extra_datasets import load_extra_dataset

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
output_dir = Path(__file__).resolve().parent / 'ml_datasets'
by_position_path = output_dir / 'rank_null_spikein_by_position.csv'
summary_path = output_dir / 'rank_null_spikein_summary.csv'

save_results = True

# when save_results and the CSVs already hold datasets not run this time, keep
# those rows and merge the new ones in (so a run of just the new datasets adds
# to the file rather than replacing it)
merge_with_saved = False

seeds = list(range(1, 51))
methods = ('rf', 'rf_one_time', 'ufi')
method_labels = {
    'rf': 'RF', 'rf_one_time': 'one-time', 'rf_per_tree': 'per-tree', 'ufi': 'UFI',
}

jitter_strength = 1e-4
n_estimators = 100
max_depth = None

# match_real: add one permuted null per real predictor (doubles p). When False,
# add exactly n_null_continuous / n_null_categorical.
match_real = True
n_null_continuous = 25
n_null_categorical = 25

dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    'forest_fires': 'Forest Fires',
    'student_performance': 'Student Performance',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'hepatitis': 'Hepatitis',
    'german_credit': 'German Credit',
    'qsar_biodeg': 'QSAR Biodegradation',
    'tokyo1': 'Tokyo1',
    'hypothyroid': 'Hypothyroid',
}

# datasets pulled straight from OpenML/PMLB with the generic recipe in
# extra_datasets.load_extra_dataset (<=20-distinct type rule, median
# imputation, missing categorical as its own level, integer coding)
extra_specs = {
    'german_credit': {'source': 'openml', 'id': 31, 'target': None,
                      'task': 'classification', 'title': 'German Credit'},
    'qsar_biodeg': {'source': 'openml', 'id': 1494, 'target': None,
                    'task': 'classification', 'title': 'QSAR Biodegradation'},
    'tokyo1': {'source': 'pmlb', 'id': 'tokyo1', 'target': None,
               'task': 'classification', 'title': 'Tokyo1'},
    'hypothyroid': {'source': 'pmlb', 'id': 'hypothyroid', 'target': None,
                    'task': 'classification', 'title': 'Hypothyroid'},
}

# datasets computed from scratch for the figure (the seven of Table 2). Other keys
# in dataset_titles are available but are not shown in the paper.
datasets = [
    'cylinder_banding', 'saheart', 'titanic', 'hepatitis',
    'german_credit', 'qsar_biodeg', 'hypothyroid',
]

kind_name = {
    0: 'real_continuous', 1: 'real_categorical',
    2: 'null_continuous', 3: 'null_categorical',
}


#----------------------------------------------------------------
# Null spike-in
#----------------------------------------------------------------
def spike_in_nulls(X, cat_idx, rng, n_cont, n_cat):
    """Append n_cont null continuous and n_cat null categorical predictors to X.

    Each null is an independent row permutation of a real column of that type,
    cycling through the real columns when more nulls are requested than exist.
    Returns (X_aug, aug_cat_idx, kind); kind[j] is 0 real continuous,
    1 real categorical, 2 null continuous, 3 null categorical.
    """
    n, p = X.shape
    categorical = set(cat_idx)
    continuous_sources = [j for j in range(p) if j not in categorical]
    categorical_sources = list(cat_idx)

    real_kind = np.ones(p, dtype=int)
    real_kind[continuous_sources] = 0
    blocks = [X]
    kinds = [real_kind]

    def permuted_block(sources, count):
        block = np.empty((n, count))
        for k in range(count):
            block[:, k] = X[rng.permutation(n), sources[k % len(sources)]]
        return block

    if n_cont and continuous_sources:
        blocks.append(permuted_block(continuous_sources, n_cont))
        kinds.append(np.full(n_cont, 2))
    if n_cat and categorical_sources:
        blocks.append(permuted_block(categorical_sources, n_cat))
        kinds.append(np.full(n_cat, 3))

    kind = np.concatenate(kinds)
    x_aug = np.hstack(blocks)
    aug_cat_idx = list(cat_idx) + list(np.where(kind == 3)[0])
    return x_aug, aug_cat_idx, kind


#----------------------------------------------------------------
# Run each importance method over every augmented dataset and seed
#----------------------------------------------------------------
from methods.jitter import jitterRF, run_ufi

jitter_config = {
    'rf': {'jitter_method': None},
    'rf_one_time': {'jitter_method': 'one_time', 'jitter_strength': jitter_strength},
    'rf_per_tree': {'jitter_method': 'per_tree', 'jitter_strength': jitter_strength},
}


def run_method(method, X, y, task, cat_idx):
    """Importance scores for one method on the augmented data."""
    if method == 'ufi':
        return run_ufi(X, y, task=task, n_estimators=n_estimators, max_depth=max_depth)
    return jitterRF(
        X, y, task=task, cat_idx=cat_idx,
        n_estimators=n_estimators, max_depth=max_depth,
        **jitter_config[method],
    )


def load_dataset(key):
    """Return (X, y, cat_idx, task) for the fully preprocessed dataset."""
    if key in extra_specs:
        x_df, y_series, cat_idx, task, *_ = load_extra_dataset(extra_specs[key])
    else:
        dataset = load_and_clean_dataset(key)
        x_df, y_series, cat_idx = preprocess_full_dataset(dataset)
        task = dataset.task
    return x_df.to_numpy(dtype=float), y_series.to_numpy(), list(cat_idx), task


rows = []
for key in datasets:
    X, y, cat_idx, task = load_dataset(key)
    n, p = X.shape
    n_categorical = len(cat_idx)
    n_continuous = p - n_categorical

    if match_real:
        n_cont, n_cat = n_continuous, n_categorical
    else:
        n_cont, n_cat = n_null_continuous, n_null_categorical

    print(f'{dataset_titles[key]}: n={n}, p={p} ({n_continuous} continuous, '
          f'{n_categorical} categorical); adding {n_cont} + {n_cat} nulls; task={task}')

    for seed in seeds:
        rng = np.random.default_rng(seed)
        x_aug, aug_cat_idx, kind = spike_in_nulls(X, cat_idx, rng, n_cont, n_cat)
        p_aug = x_aug.shape[1]
        # distinct-value count of each augmented column (permutation-invariant)
        n_unique = np.array([np.unique(x_aug[:, j]).size for j in range(p_aug)])
        tiebreak = np.random.default_rng(seed + 10_000).random(p_aug)

        for method in methods:
            np.random.seed(seed)
            result = run_method(method, x_aug, y, task, aug_cat_idx)
            scores = np.asarray(result['importances'], dtype=float).reshape(-1)
            order = np.lexsort((tiebreak, -scores))  # positions from most to least important
            for position, feature in enumerate(order, start=1):
                rows.append({
                    'dataset': key,
                    'method': method,
                    'seed': seed,
                    'rank': position,
                    'p_aug': p_aug,
                    'kind': kind_name[int(kind[feature])],
                    'n_unique': int(n_unique[feature]),
                })
        print(f'  seed {seed} done', flush=True)
    print()

by_position = pd.DataFrame(rows)

if merge_with_saved and by_position_path.exists():
    prior = pd.read_csv(by_position_path)
    prior = prior[~prior['dataset'].isin(datasets)]
    if not prior.empty:
        print(f'merging with {sorted(prior["dataset"].unique())} from {by_position_path.name}')
        by_position = pd.concat([prior.drop(columns='percentile', errors='ignore'),
                                 by_position], ignore_index=True)

by_position['percentile'] = (
    100 * (by_position['rank'] - 1) / (by_position['p_aug'] - 1)
)

#----------------------------------------------------------------
# Summary: mean percentile of each null type, per dataset x method
#----------------------------------------------------------------
null_rows = by_position[by_position['kind'].str.startswith('null')]
summary = (
    null_rows.groupby(['dataset', 'method', 'kind'])['percentile']
    .mean()
    .unstack('kind')
    .rename(columns={'null_continuous': 'cont', 'null_categorical': 'cat'})
)
summary['gap'] = summary['cont'] - summary['cat']

# datasets to show in the printout / LaTeX: everything now in the frame, in
# dataset_titles order
report_datasets = [k for k in dataset_titles if k in set(by_position['dataset'])]

if save_results:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_position.to_csv(by_position_path, index=False)
    summary.reset_index().to_csv(summary_path, index=False)
    print(f'saved {by_position_path}')
    print(f'saved {summary_path}')

#----------------------------------------------------------------
# Plain text
#----------------------------------------------------------------
print()
print(f'Mean importance-rank percentile of synthetic null predictors '
      f'(0 = most important, 100 = least), over {len(seeds)} seeds')
print('cont, cat: null continuous / null categorical.  gap = cont - cat  '
      '(negative = continuous nulls ranked more important)')
print()

name_width = 22
method_width = 12
col_width = 9
header = (
    f'{"dataset":{name_width}s}{"method":{method_width}s}'
    + ''.join(f'{c:>{col_width}s}' for c in ('cont', 'cat', 'gap'))
)
print(header)
print('-' * len(header))
for key in report_datasets:
    shown_title = dataset_titles[key]
    for method in methods:
        row = summary.loc[(key, method)]
        line = f'{shown_title:{name_width}s}{method_labels[method]:{method_width}s}'
        shown_title = ''
        for column in ('cont', 'cat', 'gap'):
            line += f'{int(round(row[column])):>{col_width}d}'
        print(line)

#----------------------------------------------------------------
# LaTeX
#----------------------------------------------------------------
print()
print('=' * 64)
print()
print(r'\begin{table}[htbp]')
print(r'\centering')
print(r'\small')
print(r'\setlength{\tabcolsep}{6pt}')
print(r'\renewcommand{\arraystretch}{1.05}')
print(r'\begin{tabular}{llccc}')
print(r'\toprule')
print(r'Dataset & Method & Null cont. & Null cat. & $\Delta$ \\')
print(r'\midrule')
for index, key in enumerate(report_datasets):
    if index > 0:
        print(r'\midrule')
    row_label = dataset_titles[key]
    for method in methods:
        row = summary.loc[(key, method)]
        cont, cat, gap = (int(round(row[c])) for c in ('cont', 'cat', 'gap'))
        print(f'{row_label} & {method_labels[method]} & {cont} & {cat} & {gap} \\\\')
        row_label = ''
print(r'\bottomrule')
print(r'\end{tabular}')
print(
    r'\caption{\textit{Importance rank of synthetic null predictors}. Mean rank '
    r'percentile ($0$ = most important, $100$ = least) of the null continuous and '
    r'null categorical predictors added to each dataset, and their difference '
    rf'$\Delta$, over {len(seeds)} random seeds. Each dataset is augmented with '
    r'one null predictor per real predictor, formed as independent row '
    r'permutations that preserve the marginals; the two null types are therefore '
    r'equivalent in everything but type, so a bias-free measure gives $\Delta=0$.}'
)
print(r'\label{tab:rank_null_spikein}')
print(r'\end{table}')
