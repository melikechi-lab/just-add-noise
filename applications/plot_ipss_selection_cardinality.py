# Cardinality of the predictors IPSS selects, by method
"""One panel per dataset. Rows are selectors; each row is a horizontal stacked bar
whose length is the number of predictors that selector picks at q <= q_threshold,
split by the predictor's number of categories. Continuous predictors fall in the
top bin (>= 21 / continuous), so the bar is the whole selected set.

Reads ipss_selection_details.json (written by ipss_selection_study.py) and the
cleaned datasets, for the per-predictor category counts.

The story: standard RF's selected set skews red (high-cardinality and continuous);
one-time jittering and UFI shift it toward yellow (low-cardinality categoricals)
and shorten it.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from clean_selected_datasets import load_and_clean_dataset, preprocess_full_dataset
from extra_datasets import QSAR_BIODEG_DESCRIPTORS, load_extra_dataset

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_fig = True

script_dir = Path(__file__).resolve().parent
details_path = script_dir / 'ml_datasets' / 'ipss_selection_details.json'
output_path = script_dir / 'ipss_selection_cardinality.png'

# a predictor counts as selected for a selector if its IPSS q-value is at or below this
q_threshold = 0.1

savefig_dpi = 300

# selectors to show, top to bottom in each panel
methods_to_show = ['RF', 'RF-jitter', 'UFI', 'GB', 'GB-jitter']
method_labels = {
    'RF': 'RF',
    'RF-jitter': 'RF-1T',
    'UFI': 'UFI',
    'GB': 'XGB',
    'GB-jitter': 'XGB-1T',
}

# the six datasets shown in the figure; ipss_selection_study.py also computes
# Forest Fires and Student Performance, which are not plotted here
dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'german_credit': 'German Credit',
    'qsar_biodeg': 'QSAR Biodegradation',
    'hypothyroid': 'Hypothyroid',
}

extra_specs = {
    'german_credit': {'source': 'openml', 'id': 31, 'target': None,
                      'task': 'classification', 'title': 'German Credit'},
    'qsar_biodeg': {'source': 'openml', 'id': 1494, 'target': None,
                    'task': 'classification', 'title': 'QSAR Biodegradation',
                    'rename': QSAR_BIODEG_DESCRIPTORS},
    'hypothyroid': {'source': 'pmlb', 'id': 'hypothyroid', 'target': None,
                    'task': 'classification', 'title': 'Hypothyroid'},
}

# (low, high, label); a categorical with `low <= n_categories <= high` lands here,
# continuous predictors land in the last bin. Colours run yellow -> red.
cardinality_bins = [
    (2, 2, '2'),
    (3, 4, '3-4'),
    (5, 8, '5-8'),
    (9, 20, '9-20'),
    (21, np.inf, '>20/continuous'),
]
bin_colors = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(cardinality_bins)))

#----------------------------------------------------------------
# Load saved results
#----------------------------------------------------------------
if not details_path.exists():
    raise SystemExit(f'{details_path} not found; run ipss_selection_study.py first')

with open(details_path) as handle:
    details = json.load(handle)


def selected(key, method, group):
    return [e for e in details[key][method][group] if e['q_value'] <= q_threshold]


#----------------------------------------------------------------
# Per-predictor category counts (same loaders as ipss_selection_study.py)
#----------------------------------------------------------------
def dataset_info(key):
    if key in extra_specs:
        x_df, _, cat_idx, *_ = load_extra_dataset(extra_specs[key])
    else:
        x_df, _, cat_idx = preprocess_full_dataset(load_and_clean_dataset(key))
    columns = list(x_df.columns)
    return {
        'levels': {columns[j]: int(x_df.iloc[:, j].nunique()) for j in cat_idx},
        'p': x_df.shape[1],
        'n_categorical': len(cat_idx),
    }


info = {key: dataset_info(key) for key in dataset_titles}

# panels ordered by number of predictors, most to least
datasets = sorted(dataset_titles, key=lambda key: info[key]['p'], reverse=True)


def bin_index(n_categories):
    for i, (low, high, _) in enumerate(cardinality_bins):
        if low <= n_categories <= high:
            return i
    return len(cardinality_bins) - 1


def counts_by_bin(key, method):
    """Number of selected predictors in each cardinality bin, for one selector."""
    counts = np.zeros(len(cardinality_bins), dtype=int)
    for entry in selected(key, method, 'categorical'):
        counts[bin_index(info[key]['levels'][entry['name']])] += 1
    counts[-1] += len(selected(key, method, 'continuous'))
    return counts


#----------------------------------------------------------------
# Figure
#----------------------------------------------------------------
n_cols = 2
n_rows = int(np.ceil(len(datasets) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), layout='constrained')
axes = axes.flatten()

y = np.arange(len(methods_to_show))
for ax, key in zip(axes, datasets):
    grid = np.vstack([counts_by_bin(key, method) for method in methods_to_show])
    left = np.zeros(len(methods_to_show))
    for i, color in enumerate(bin_colors):
        ax.barh(y, grid[:, i], left=left, color=color, edgecolor='white', linewidth=0.5)
        left += grid[:, i]
    for yi, total in zip(y, left):
        ax.text(total + 0.15, yi, str(int(total)), va='center', fontsize=9, color='0.3')

    ax.set_yticks(y)
    ax.set_yticklabels([method_labels[m] for m in methods_to_show])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_title(
        f'{dataset_titles[key]}  ({info[key]["p"] - info[key]["n_categorical"]} continuous, '
        f'{info[key]["n_categorical"]} categorical)',
        fontsize=10, loc='left',
    )
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)

for ax in axes[len(datasets):]:
    ax.set_visible(False)

# fig.supxlabel('Number of predictors selected', fontsize=12)
fig.legend(
    handles=[Patch(facecolor=c, label=lab) for c, (_, _, lab) in zip(bin_colors, cardinality_bins)],
    loc='outside lower center', ncol=len(cardinality_bins), frameon=False,
    fontsize=14, handlelength=1.8, handleheight=1.4, columnspacing=1.6, labelspacing=0.4,
)

if save_fig:
    fig.savefig(output_path, dpi=savefig_dpi, bbox_inches='tight')
    print(f'saved {output_path}')

plt.show()
