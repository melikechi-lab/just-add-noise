# IPSS null spike-in study for the ML datasets
"""Add synthetic null predictors to each ML dataset and count how many each IPSS
selector picks up.

The nulls are permuted copies of real columns: a null continuous predictor is an
independent row permutation of a real continuous column---same marginal, no
association with y---and likewise for null categorical predictors. Because the
nulls are inactive by construction, the number selected is a direct read on each
selector's tendency toward false positives of that feature type, the continuous
side being exactly the bias this paper targets. With `match_real` (the default),
each draw adds one null per real predictor, so the augmented dataset has twice as
many predictors as the original.

For each dataset we draw `n_draws` independent sets of nulls, run IPSS with each
selector at `target_fdr`, and average the counts over draws. Running all five
selectors over ten draws on five datasets is a few hundred IPSS fits; trim
`methods` or `n_draws` to shorten it.

Outputs (ml_datasets/):
  ipss_null_spikein_by_draw.csv   one row per dataset x selector x draw
  ipss_null_spikein_summary.csv   one row per dataset x selector (mean, sd)
Then prints a plain-text summary and a LaTeX table.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_selected_datasets import load_and_clean_dataset, preprocess_full_dataset
from extra_datasets import load_extra_dataset

try:
    from ipss import ipss
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f'this script requires the ipss package ({exc})')

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
output_dir = Path(__file__).resolve().parent / 'ml_datasets'
by_draw_path = output_dir / 'ipss_null_spikein_by_draw.csv'
summary_path = output_dir / 'ipss_null_spikein_summary.csv'

save_results = True
target_fdr = 0.1
random_seed = 302
jitter_strength = 1e-4
n_jobs = -1

# match_real: add one permuted null per real predictor, so each augmented dataset
# has twice as many predictors as the original. When False, add exactly
# n_null_continuous / n_null_categorical nulls regardless of the dataset.
match_real = True
n_null_continuous = 25   # used only when match_real is False
n_null_categorical = 25  # used only when match_real is False
n_draws = 10             # independent null draws to average over

show_sd = True           # plain-text: show (sd) next to each mean

# selectors to run; same configuration as ipss_selection_study.py
methods = {
    'RF':        {'selector': 'rf',  'jitter': False, 'delta': None, 'B': 100},
    'RF-jitter': {'selector': 'rf',  'jitter': True,  'delta': None, 'B': 100},
    'GB':        {'selector': 'gb',  'jitter': False, 'delta': None, 'B': 100},
    'GB-jitter': {'selector': 'gb',  'jitter': True,  'delta': None, 'B': 100},
    'UFI':       {'selector': 'ufi', 'jitter': False, 'delta': None, 'B': 100},
}

dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'hepatitis': 'Hepatitis',
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
                    'task': 'classification', 'title': 'QSAR Biodegradation'},
    'hypothyroid': {'source': 'pmlb', 'id': 'hypothyroid', 'target': None,
                    'task': 'classification', 'title': 'Hypothyroid'},
}

# datasets to (re)compute this run; when merge_with_saved, rows for the other
# datasets in dataset_titles are kept from the saved CSV
# compute every dataset from scratch; set merge_with_saved = True and a shorter
# `datasets` list to update only some rows against an existing run.
datasets = list(dataset_titles)
merge_with_saved = False


def load_dataset(key):
    """Return (X, y, cat_idx) for the fully preprocessed dataset."""
    if key in extra_specs:
        x_df, y_series, cat_idx, *_ = load_extra_dataset(extra_specs[key])
    else:
        dataset = load_and_clean_dataset(key)
        x_df, y_series, cat_idx = preprocess_full_dataset(dataset)
    return x_df.to_numpy(dtype=float), y_series.to_numpy(), list(cat_idx)


#----------------------------------------------------------------
# Null spike-in
#----------------------------------------------------------------
def spike_in_nulls(X, cat_idx, rng, n_cont, n_cat):
    """Append n_cont null continuous and n_cat null categorical predictors to X.

    Each null is an independent row permutation of a real column of that type,
    cycling through the real columns when more nulls are requested than exist.
    Returns (X_aug, aug_cat_idx, kind), where kind[j] is 0 for a real column, 1
    for a null continuous column, and 2 for a null categorical column.
    """
    n, p = X.shape
    categorical = set(cat_idx)
    continuous_sources = [j for j in range(p) if j not in categorical]
    categorical_sources = list(cat_idx)

    blocks = [X]
    kinds = [np.zeros(p, dtype=int)]

    def permuted_block(sources, count):
        block = np.empty((n, count))
        for k in range(count):
            block[:, k] = X[rng.permutation(n), sources[k % len(sources)]]
        return block

    if n_cont and continuous_sources:
        blocks.append(permuted_block(continuous_sources, n_cont))
        kinds.append(np.full(n_cont, 1))
    if n_cat and categorical_sources:
        blocks.append(permuted_block(categorical_sources, n_cat))
        kinds.append(np.full(n_cat, 2))

    kind = np.concatenate(kinds)
    x_aug = np.hstack(blocks)
    aug_cat_idx = list(cat_idx) + list(np.where(kind == 2)[0])
    return x_aug, aug_cat_idx, kind


def jitter_categoricals(X, cat_idx, rng):
    """One uniform jitter realization on the categorical columns."""
    out = X.astype(float, copy=True)
    out[:, cat_idx] += rng.uniform(
        -jitter_strength, jitter_strength, size=(X.shape[0], len(cat_idx))
    )
    return out


#----------------------------------------------------------------
# Run IPSS on every augmented dataset
#----------------------------------------------------------------
rows = []
for key in datasets:
    X, y, cat_idx = load_dataset(key)
    n, p = X.shape
    n_categorical = len(cat_idx)
    n_continuous = p - n_categorical
    real_categorical = set(cat_idx)

    if match_real:
        n_cont, n_cat = n_continuous, n_categorical
    else:
        n_cont, n_cat = n_null_continuous, n_null_categorical
    print(f'{dataset_titles[key]}: n={n}, p={p} '
          f'({n_continuous} continuous, {n_categorical} categorical); '
          f'adding {n_cont} + {n_cat} nulls')

    for draw in range(n_draws):
        rng = np.random.default_rng(random_seed + draw)
        x_aug, aug_cat_idx, kind = spike_in_nulls(X, cat_idx, rng, n_cont, n_cat)
        x_aug_jittered = jitter_categoricals(
            x_aug, aug_cat_idx, np.random.default_rng(random_seed + 10_000 + draw)
        )

        for name, config in methods.items():
            np.random.seed(random_seed + draw)
            x_input = x_aug_jittered if config['jitter'] else x_aug
            kwargs = {
                'selector': config['selector'],
                'target_fdr': target_fdr,
                'n_jobs': n_jobs,
                'B': config['B'],
            }
            if config['delta'] is not None:
                kwargs['delta'] = config['delta']

            result = ipss(x_input, y, **kwargs)
            selected = {int(i) for i in result['selected_features']}
            rows.append({
                'dataset': key,
                'selector': name,
                'draw': draw,
                'null_continuous_added': n_cont,
                'null_categorical_added': n_cat,
                'null_continuous_selected': int(sum(kind[i] == 1 for i in selected)),
                'null_categorical_selected': int(sum(kind[i] == 2 for i in selected)),
                'real_selected': int(sum(kind[i] == 0 for i in selected)),
                'real_categorical_selected': int(
                    sum(kind[i] == 0 and i in real_categorical for i in selected)
                ),
            })
        print(f'  draw {draw + 1}/{n_draws} done', flush=True)
    print()

by_draw = pd.DataFrame(rows)

if merge_with_saved and by_draw_path.exists():
    prior = pd.read_csv(by_draw_path)
    keep = prior['dataset'].isin(dataset_titles) & ~prior['dataset'].isin(datasets)
    prior = prior[keep]
    if not prior.empty:
        print(f'merging with {sorted(prior["dataset"].unique())} from {by_draw_path.name}')
        by_draw = pd.concat([prior, by_draw], ignore_index=True)

# report / summary cover every dataset now in the frame, in dataset_titles order
report_datasets = [k for k in dataset_titles if k in set(by_draw['dataset'])]

count_columns = [
    'null_continuous_selected',
    'null_categorical_selected',
    'real_selected',
    'real_categorical_selected',
]
summary = by_draw.groupby(['dataset', 'selector'], sort=False)[count_columns].agg(
    ['mean', 'std']
)

# null-counts added per dataset (from this run and from any merged-in rows)
added = {
    key: (int(g['null_continuous_added'].iloc[0]), int(g['null_categorical_added'].iloc[0]))
    for key, g in by_draw.groupby('dataset')
}

if save_results:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_draw.to_csv(by_draw_path, index=False)
    flat = summary.copy()
    flat.columns = ['_'.join(parts) for parts in flat.columns]
    flat.reset_index().to_csv(summary_path, index=False)
    print(f'saved {by_draw_path}')
    print(f'saved {summary_path}')

#----------------------------------------------------------------
# Plain-text summary
#----------------------------------------------------------------
selector_names = list(methods)


def cell(dataset, selector, column):
    mean = summary.loc[(dataset, selector), (column, 'mean')]
    sd = summary.loc[(dataset, selector), (column, 'std')]
    if pd.isna(sd):
        sd = 0.0
    text = f'{mean:.1f} ({sd:.1f})' if show_sd else f'{mean:.1f}'
    if column == 'null_continuous_selected':
        text += f' / {added[dataset][0]}'
    elif column == 'null_categorical_selected':
        text += f' / {added[dataset][1]}'
    return text


print()
print(f'Synthetic null predictors selected by IPSS at target FDR {target_fdr}')
if match_real:
    print(f'one null per real predictor (doubling p); mean (sd) over {n_draws} draws')
else:
    print(f'{n_null_continuous} null continuous and {n_null_categorical} null '
          f'categorical predictors added; mean (sd) over {n_draws} draws')
print('null columns shown as  selected / added')
print()

name_width = 22
sel_width = 12
col_width = 18
header = (
    f'{"dataset":{name_width}s}{"selector":{sel_width}s}'
    f'{"null cont.":>{col_width}s}{"null cat.":>{col_width}s}'
    f'{"real sel.":>{col_width}s}{"real cat.":>{col_width}s}'
)
print(header)
print('-' * len(header))
for key in report_datasets:
    shown_title = dataset_titles[key]
    for selector in selector_names:
        line = f'{shown_title:{name_width}s}{selector:{sel_width}s}'
        shown_title = ''
        for column in count_columns:
            line += f'{cell(key, selector, column):>{col_width}s}'
        print(line)

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
print(r'\renewcommand{\arraystretch}{1.05}')
print(r'\begin{tabular}{llccc}')
print(r'\toprule')
print(r'Dataset & Selector & Null continuous & Null categorical & Real \\')
print(r'\midrule')
for index, key in enumerate(report_datasets):
    if index > 0:
        print(r'\midrule')
    shown_title = f'{dataset_titles[key]} ({added[key][0]}$+${added[key][1]})'
    for selector in selector_names:
        cont = summary.loc[(key, selector), ('null_continuous_selected', 'mean')]
        cat = summary.loc[(key, selector), ('null_categorical_selected', 'mean')]
        real = summary.loc[(key, selector), ('real_selected', 'mean')]
        print(f'{shown_title} & {selector} & {cont:.1f} & {cat:.1f} & {real:.1f} \\\\')
        shown_title = ''
print(r'\bottomrule')
print(r'\end{tabular}')
if match_real:
    added_clause = (
        r'each draw adds one null predictor per real predictor---an independent '
        r'row permutation of that predictor---doubling the predictor count '
        r'(null continuous $+$ null categorical added, shown after each dataset name)'
    )
else:
    added_clause = (
        rf'each draw adds ${n_null_continuous}$ null continuous and '
        rf'${n_null_categorical}$ null categorical predictors, formed as '
        r'independent row permutations of real columns'
    )
print(
    r'\caption{\textit{Synthetic null predictors selected by IPSS}. Mean number '
    r'of synthetic null predictors selected by each base selector at target FDR '
    rf'$\alpha={target_fdr}$, over ${n_draws}$ independent draws; ' + added_clause + r'. '
    r'\textit{Real} is the mean number of real predictors selected, essentially '
    r'unchanged from the un-augmented data.}'
)
print(r'\label{tab:ipss_null_spikein}')
print(r'\end{table}')
