# Rank box-and-whisker plots by method
"""Box-and-whisker plot of a variable's importance rank across seeds, per method.

Reads the by-seed importance table written by
bladder_18m_survival_mirna_ordinal_stage.py and, for one chosen variable
(pathologic_stage by default), draws a box per method summarizing how that
variable's rank varies across random seeds. Lower ranks are more important.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_fig = True

data_dir = Path(__file__).resolve().parent / 'data' / 'bladder'
data_path = data_dir / 'bladder_18m_survival_mirna_ordinal_stage_importance_by_seed.csv'
output_path = data_dir / 'bladder_18m_survival_mirna_ordinal_stage_rank_boxplots.png'

# variable whose per-seed rank is summarized
target_variable = 'pathologic_stage'

# methods to plot, top to bottom, using the column prefixes in the data; trim
# this list to plot a subset
methods_to_plot = [
    'rf',
    'rf_one_time',
    'rf_per_tree',
    # 'xgb',
    # 'xgb_one_time',
    # 'xgb_stump',
    # 'xgb_stump_one_time',
    'ufi',
    'cforest',
]

# display names for the axis ticks; methods without an entry use their prefix
method_labels = {
    'rf': 'RF',
    'rf_one_time': 'RF-one-time',
    'rf_per_tree': 'RF-per-tree',
    'xgb': 'XGB',
    'xgb_one_time': 'XGB one-time',
    'xgb_stump': 'XGB stump',
    'xgb_stump_one_time': 'XGB stump one-time',
    'ufi': 'UFI',
    'cforest': 'CForest',
}

fig_width = 12
fig_height = 4
savefig_dpi = 300
axis_label_fontsize = 16
x_tick_label_fontsize = 12
y_tick_label_fontsize = 13
median_linewidth = 2

#----------------------------------------------------------------
# Load the per-seed ranks
#----------------------------------------------------------------
if not methods_to_plot:
    raise ValueError('methods_to_plot is empty')

ranks = pd.read_csv(data_path)
variable_ranks = ranks.loc[ranks['variable'] == target_variable]
if variable_ranks.empty:
    raise ValueError(f'{target_variable!r} not found in {data_path}')

missing = [
    f'{method}_rank'
    for method in methods_to_plot
    if f'{method}_rank' not in variable_ranks.columns
]
if missing:
    raise ValueError(f'Missing rank columns in {data_path}: {missing}')

rank_by_method = [
    variable_ranks[f'{method}_rank'].to_numpy() for method in methods_to_plot
]
n_seeds = variable_ranks.shape[0]
n_variables = int(ranks['variable'].nunique())

print(f'Variable: {target_variable}')
print(f'Seeds: {n_seeds}, variables ranked per seed: {n_variables}')
print(f'{"method":<22} {"median":>7} {"q1":>6} {"q3":>6} {"iqr":>6} {"min":>5} {"max":>5}')
for method, values in zip(methods_to_plot, rank_by_method):
    q1, q3 = np.percentile(values, [25, 75])
    print(
        f'  {method:<20} {np.median(values):7.1f} {q1:6.1f} {q3:6.1f} '
        f'{q3 - q1:6.1f} {int(values.min()):5d} {int(values.max()):5d}'
    )

#----------------------------------------------------------------
# Box-and-whisker plot
#----------------------------------------------------------------
labels = [method_labels.get(method, method) for method in methods_to_plot]
positions = np.arange(len(methods_to_plot))

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.boxplot(
    rank_by_method,
    positions=positions,
    vert=False,
    widths=0.5,
    medianprops={'linewidth': median_linewidth},
)

ax.set_yticks(positions)
ax.set_yticklabels(labels, fontsize=y_tick_label_fontsize)
ax.invert_yaxis()
ax.set_xlim(left=0)
ax.set_xlabel(f'Rank of pathologic stage', fontsize=axis_label_fontsize)
ax.tick_params(axis='x', labelsize=x_tick_label_fontsize)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()

if save_fig:
    fig.savefig(output_path, dpi=savefig_dpi, bbox_inches='tight')
    print(f'\nSaved figure: {output_path}')

plt.show()
