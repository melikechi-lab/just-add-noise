# Where synthetic null predictors land in the importance ranking
"""One panel per dataset. Rows are importance methods, columns are importance-rank
positions (1 = most important). Colour encodes, at each rank position, what kind
of predictor sits there across seeds -- see color_mode below.

Reads rank_null_spikein_by_position.csv (run rank_null_spikein_study.py with
save_results = True first).

color_mode:
  'composition'    each cell blends category colours weighted by how often the
                   predictor at that rank is of each category. composition_split
                   '4way' uses real/null x continuous/categorical (blue / green /
                   red / yellow) and shows the mechanism -- under standard RF the
                   top mixes real and null *continuous*; '2way' collapses to real
                   (blue) vs null (red) and shows how much of the ranking's top is
                   provable noise.
  'null_continuous' cell = fraction of seeds the predictor at this rank is a
                   synthetic null continuous predictor. Under standard RF a bright
                   band sits just below the real signal; jittering and UFI push it
                   to the far right or thin it out. 'null_categorical',
                   'real_continuous', 'real_categorical' are the analogues.
  'null_fraction'  cell = fraction of seeds the predictor at this rank is a
                   synthetic null of either type. Rises with rank under any
                   method (half the columns are null), so it under-shows the bias.
  'null_type'      cell = F_cont(r) - F_cat(r), the gap between the cumulative
                   distributions of null-continuous and null-categorical ranks
                   (count-invariant). A positive (red) bulge means the null
                   continuous predictors rank more important than the null
                   categorical ones. Large under standard RF; jittering flattens
                   it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_fig = True

data_path = Path(__file__).resolve().parent / 'ml_datasets' / 'rank_null_spikein_by_position.csv'
# saved straight into the manuscript's figures folder
output_path = (
    Path(__file__).resolve().parents[3] / 'overleaf' / 'figures'
    / 'rank_null_spikein_heatmap.png'
)

color_mode = 'composition'     # composition | null_continuous | null_categorical
                               #  | real_continuous | real_categorical | null_fraction | null_type

# composition mode only:
#   '4way'             real/null x continuous/categorical, four colours
#   '3way'             real vs null continuous vs null categorical
#   '2way'             real vs null, two colours
#   'highlight'        one category (highlight_category) bright, everything else dark
#   'null_cardinality' reals a flat background, each null coloured by its number of
#                      distinct values on a yellow->red scale (binary -> >20)
composition_split = 'null_cardinality'
highlight_category = 'null_continuous'
highlight_color = 'red'          # the highlighted category
highlight_other_color = 'lightblue'    # everything else

null_cardinality_cmap = 'autumn_r'   # null_cardinality split: binary -> high-cardinality
cardinality_cap = 21              # clip the scale here (continuous nulls sit at the top)

# rescale every panel's x-axis to rank / p in [0, 1] so panels with different
# predictor counts align; False keeps the raw 1..p axis
normalize_rank = True
normalized_bins = 120

methods_to_plot = ['rf', 'rf_one_time', 'ufi']
method_labels = {
    'rf': 'RF', 'rf_one_time': 'RF-1T', 'rf_per_tree': 'per-tree', 'ufi': 'UFI',
}

dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    # 'forest_fires': 'Forest Fires',
    # 'student_performance': 'Student Performance',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'hepatitis': 'Hepatitis',
    'german_credit': 'German Credit',
    'qsar_biodeg': 'QSAR Biodegradation',
    # 'tokyo1': 'Tokyo1',
    'hypothyroid': 'Hypothyroid',
}

type_cmap = 'bwr'
fraction_cmap = 'bwr'
sequential_cmap = 'viridis'    # null_continuous / null_categorical / real_* modes
sequential_vmax = 0.6          # clip the sequential scale here for contrast

# composition mode: one colour per predictor category
category_colors = {
    'real_continuous': 'dodgerblue',    # blue
    'real_categorical': 'limegreen',   # green
    'null_continuous': 'red',    # red
    'null_categorical': 'cyan',   # yellow
    'real': 'gainsboro',               # 2way / 3way
    'null': 'red',                    # 2way
}

savefig_dpi = 300

#----------------------------------------------------------------
# Load
#----------------------------------------------------------------
if not data_path.exists():
    raise SystemExit(
        f'{data_path} not found; run rank_null_spikein_study.py with '
        f'save_results = True first'
    )

frame = pd.read_csv(data_path)
n_seeds = frame['seed'].nunique()

# panels ordered by number of real predictors, most to least
def _p_real(key):
    return int(frame.loc[frame['dataset'] == key, 'p_aug'].iloc[0]) // 2

datasets = sorted(
    (key for key in dataset_titles if key in set(frame['dataset'])),
    key=_p_real, reverse=True,
)

is_null = lambda kinds: np.isin(kinds, ('null_continuous', 'null_categorical'))

if composition_split == '4way':
    category_order = ('real_continuous', 'real_categorical',
                      'null_continuous', 'null_categorical')
    color_matrix = np.array([to_rgb(category_colors[c]) for c in category_order])
elif composition_split == '3way':
    category_order = ('real', 'null_continuous', 'null_categorical')
    color_matrix = np.array([to_rgb(category_colors[c]) for c in category_order])
elif composition_split == '2way':
    category_order = ('real', 'null')
    color_matrix = np.array([to_rgb(category_colors[c]) for c in category_order])
elif composition_split == 'highlight':
    category_order = (highlight_category, 'other')
    color_matrix = np.array([to_rgb(highlight_color), to_rgb(highlight_other_color)])
elif composition_split == 'null_cardinality':
    category_order = None  # rank_profile returns RGB directly for this split
    color_matrix = None
else:
    raise ValueError(f'unknown composition_split {composition_split!r}')

# rank_profile returns a per-rank RGB row (not category weights) for this split
rgb_profile = color_mode == 'composition' and composition_split == 'null_cardinality'


def to_group(kinds):
    """Map the raw kind labels to the composition categories."""
    if composition_split == '2way':
        return np.where(is_null(kinds), 'null', 'real')
    if composition_split == '3way':
        return np.where(is_null(kinds), kinds, 'real')
    if composition_split == 'highlight':
        return np.where(kinds == highlight_category, highlight_category, 'other')
    return kinds


indicator = {
    'null_fraction': is_null,
    'null_continuous': lambda kinds: kinds == 'null_continuous',
    'null_categorical': lambda kinds: kinds == 'null_categorical',
    'real_continuous': lambda kinds: kinds == 'real_continuous',
    'real_categorical': lambda kinds: kinds == 'real_categorical',
}

if color_mode == 'composition':
    vmin = vmax = cmap = None
elif color_mode == 'null_type':
    vmin, vmax, cmap = -0.75, 0.75, type_cmap
elif color_mode == 'null_fraction':
    vmin, vmax, cmap = 0.0, 1.0, fraction_cmap
elif color_mode in indicator:
    vmin, vmax, cmap = 0.0, sequential_vmax, sequential_cmap
else:
    raise ValueError(f'unknown color_mode {color_mode!r}')


#----------------------------------------------------------------
# Per-rank profile, per method and dataset
#----------------------------------------------------------------
def rank_profile(sub, method):
    """One row of the heatmap for a method, on the normalized or raw rank axis:
    the CDF gap (null_type) or the mean per-rank indicator (the other modes)."""
    method_rows = sub[sub['method'] == method]
    p_aug = int(method_rows['p_aug'].iloc[0])
    raw_axis = 100 * (np.arange(1, p_aug + 1) - 1) / (p_aug - 1)
    axis = np.linspace(0, 100, normalized_bins) if normalize_rank else raw_axis

    if color_mode == 'null_type':
        cont = np.sort(method_rows.loc[method_rows['kind'] == 'null_continuous', 'percentile'])
        cat = np.sort(method_rows.loc[method_rows['kind'] == 'null_categorical', 'percentile'])
        f_cont = np.searchsorted(cont, axis, side='right') / len(cont)
        f_cat = np.searchsorted(cat, axis, side='right') / len(cat)
        return f_cont - f_cat

    seeds = method_rows['seed'].unique()
    by_seed = [
        method_rows[method_rows['seed'] == seed].sort_values('rank')
        for seed in seeds
    ]
    by_seed_kinds = [rows_seed['kind'].to_numpy() for rows_seed in by_seed]

    def _to_axis(array_2d):
        if not normalize_rank:
            return array_2d
        return np.column_stack(
            [np.interp(axis, raw_axis, array_2d[:, c]) for c in range(array_2d.shape[1])]
        )

    if rgb_profile:
        background = np.array(to_rgb(category_colors['real']))
        cmap_fn = plt.get_cmap(null_cardinality_cmap)
        acc = np.zeros((p_aug, 3))
        for rows_seed in by_seed:
            kinds = rows_seed['kind'].to_numpy()
            n_unique = rows_seed['n_unique'].to_numpy()
            colours = np.tile(background, (p_aug, 1))
            mask = is_null(kinds)
            t = (np.clip(n_unique[mask], 2, cardinality_cap) - 2) / (cardinality_cap - 2)
            colours[mask] = cmap_fn(t)[:, :3]
            acc += colours
        return _to_axis(acc / len(seeds))

    if color_mode == 'composition':
        probs = np.zeros((p_aug, len(category_order)))
        for kinds in by_seed_kinds:
            groups = to_group(kinds)
            for ci, category in enumerate(category_order):
                probs[:, ci] += groups == category
        probs /= len(seeds)
        return _to_axis(probs)

    is_target = indicator[color_mode]
    profile = np.zeros(p_aug)
    for kinds in by_seed_kinds:
        profile += is_target(kinds)
    profile /= len(seeds)
    if normalize_rank:
        profile = np.interp(axis, raw_axis, profile)
    return profile


panels = []
for key in datasets:
    sub = frame[frame['dataset'] == key]
    available = [m for m in methods_to_plot if m in set(sub['method'])]
    first = sub[(sub['method'] == available[0]) & (sub['seed'] == sub['seed'].iloc[0])]
    counts = first['kind'].value_counts()
    p_real = int(first['p_aug'].iloc[0]) // 2
    profiles = [rank_profile(sub, method) for method in available]
    grid = np.stack(profiles) if color_mode == 'composition' else np.vstack(profiles)
    panels.append({
        'title': dataset_titles[key],
        'methods': available,
        'grid': grid,
        'p_real': p_real,
        'n_continuous': int(counts.get('real_continuous', 0)),
        'n_categorical': int(counts.get('real_categorical', 0)),
    })

#----------------------------------------------------------------
# Figure
#----------------------------------------------------------------
n_panels = len(panels)
fig, axes = plt.subplots(
    n_panels, 1,
    figsize=(10,9),
    layout='constrained',
)
if n_panels == 1:
    axes = [axes]

image = None
for ax, panel in zip(axes, panels):
    grid = panel['grid']
    n_methods, n_cols = grid.shape[:2]
    x_hi = 1.0 if normalize_rank else n_cols
    extent = [0.0 if normalize_rank else 0.5, x_hi + (0 if normalize_rank else 0.5),
              n_methods - 0.5, -0.5]
    if rgb_profile:
        ax.imshow(np.clip(grid, 0.0, 1.0), aspect='auto', extent=extent)
    elif color_mode == 'composition':
        ax.imshow(np.clip(grid @ color_matrix, 0.0, 1.0), aspect='auto', extent=extent)
    else:
        image = ax.imshow(
            grid, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
        )
    if color_mode != 'null_type':
        boundary = 0.5 if normalize_rank else panel['p_real'] + 0.5
        ax.axvline(boundary, color='0.6', lw=1, ls=(0, (4, 2)))
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([method_labels.get(m, m) for m in panel['methods']])
    ax.set_title(
        f'{panel["title"]} '
        f'({panel["n_continuous"]} continuous, {panel["n_categorical"]} categorical)',
        fontsize=10, loc='left',
    )
    ax.tick_params(length=0, labelbottom=(ax is axes[-1]))
    for spine in ax.spines.values():
        spine.set_visible(False)

# axes[-1].set_xlabel(
#     'importance rank / p' if normalize_rank else 'importance rank (1 = most important)'
# )

if rgb_profile:
    scale = ScalarMappable(cmap=null_cardinality_cmap, norm=Normalize(2, cardinality_cap))
    cbar = fig.colorbar(
        scale, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04, aspect=45
    )
    cbar.set_label('Number of categories (synthetic null predictors only)', fontsize=14)
    cbar.set_ticks([2, 5, 10, 15, cardinality_cap])
    cbar.set_ticklabels(['2', '5', '10', '15', f'{cardinality_cap - 1}+'])
    suptitle = f'Null-predictor cardinality by importance rank, over {n_seeds} seeds'
elif color_mode == 'composition':
    # 'real' is the neutral background in 2way / 3way; leave it out of the legend
    legend_entries = [
        (color_matrix[i], c.replace('_', ' ') if c != 'other' else 'other predictors')
        for i, c in enumerate(category_order)
        if c != 'real'
    ]
    fig.legend(
        handles=[Patch(facecolor=colour, label=label) for colour, label in legend_entries],
        loc='outside lower center', ncol=len(legend_entries), frameon=False,
    )
    suptitle = f'Predictor composition by importance rank, over {n_seeds} seeds'
else:
    cbar = fig.colorbar(
        image, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04, aspect=45
    )
    if color_mode == 'null_type':
        cbar.set_label(
            r'$F_{\mathrm{cont}}(r) - F_{\mathrm{cat}}(r)$: '
            'null continuous predictors ranked more important ($+$) or less ($-$)'
        )
        cbar.set_ticks([-0.75, 0, 0.75])
        suptitle = f'Null-continuous vs null-categorical rank, over {n_seeds} seeds'
    elif color_mode == 'null_fraction':
        cbar.set_label(
            'fraction of seeds the predictor at this rank is a synthetic null '
            '(ideal: pale left of the dashed line, dark to its right)'
        )
        cbar.set_ticks([0, 0.5, 1])
        suptitle = f'How far up the ranking synthetic nulls reach, over {n_seeds} seeds'
    else:
        label = color_mode.replace('_', ' ')
        cbar.set_label(
            f'fraction of seeds the predictor at this rank is a {label} predictor'
        )
        cbar.set_ticks([0, sequential_vmax / 2, sequential_vmax])
        suptitle = f'Where {label} predictors rank, over {n_seeds} seeds'

# fig.suptitle(suptitle, fontsize=12)

# constrained_layout handles spacing; tight_layout is incompatible with the
# figure-level colorbar.

if save_fig:
    fig.savefig(output_path, dpi=savefig_dpi, bbox_inches='tight')
    print(f'saved {output_path}')

plt.show()
