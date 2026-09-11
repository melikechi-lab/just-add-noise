# LaTeX and plain-text table for the IPSS null spike-in results
"""Re-tabulate ml_datasets/ipss_null_spikein_by_draw.csv (written by
ipss_null_spikein_study.py) without re-running the study.

Two layouts:

  'by_dataset'   one row per dataset; columns n, p, q, then one per selector.
                 p and q are the numbers of continuous and categorical
                 predictors (equivalently, the numbers of null continuous and
                 null categorical predictors added, one per real predictor).
                 Each selector cell is the mean number of nulls selected, with
                 the number that are continuous nulls in parentheses.

  'by_selector'  one row per (dataset, selector); columns for null continuous,
                 null categorical, and optionally the real selections.

With `as_percent`, counts become percentages of the number available (per type).

Prints plain text, then a LaTeX table.
"""

from pathlib import Path

import pandas as pd

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
data_dir = Path(__file__).resolve().parent / 'ml_datasets'
by_draw_path = data_dir / 'ipss_null_spikein_by_draw.csv'

layout = 'by_dataset'          # 'by_dataset' or 'by_selector'

# selectors to show, in order; None shows every selector in the file, in order
selectors_to_show = ['RF', 'RF-jitter', 'GB', 'GB-jitter']
selector_display = {
    'RF': 'RF',
    'RF-jitter': 'RF-one-time',
    'GB': 'XGB',
    'GB-jitter': 'XGB-one-time',
    'UFI': 'UFI',
}

as_percent = False            # counts -> percentages of the number available
percent_decimals = 0
decimals = 1

# 'by_selector' only
show_real = False              # include a 'Real' column (all original predictors)
show_real_categorical = False  # include a 'Real cat.' column
plain_text_sd = True
latex_sd = False

target_fdr_label = 0.1         # caption text only

dataset_titles = {
    'cylinder_banding': 'Cylinder Banding',
    'saheart': 'SAheart',
    'titanic': 'Titanic',
    'hepatitis': 'Hepatitis',
    'german_credit': 'German Credit',
    'qsar_biodeg': 'QSAR Biodegradation',
    'hypothyroid': 'Hypothyroid',
}

# sample sizes of the cleaned datasets (not stored in the CSV)
dataset_n = {
    'cylinder_banding': 540,
    'saheart': 462,
    'titanic': 2207,
    'hepatitis': 155,
    'german_credit': 1000,
    'qsar_biodeg': 1055,
    'hypothyroid': 3163,
}

latex_label = 'tab:ipss_null_spikein'

#----------------------------------------------------------------
# Load and aggregate
#----------------------------------------------------------------
if not by_draw_path.exists():
    raise SystemExit(f'{by_draw_path} not found; run ipss_null_spikein_study.py first')

frame = pd.read_csv(by_draw_path)
frame['null_total_selected'] = (
    frame['null_continuous_selected'] + frame['null_categorical_selected']
)
n_draws = frame['draw'].nunique()

datasets = [key for key in dataset_titles if key in set(frame['dataset'])]
for key in frame['dataset'].unique():
    if key not in dataset_titles:
        datasets.append(key)
        dataset_titles[key] = key

file_selectors = list(dict.fromkeys(frame['selector']))
selectors = [s for s in (selectors_to_show or file_selectors) if s in set(file_selectors)]

added = (
    frame.groupby('dataset')[['null_continuous_added', 'null_categorical_added']]
    .first()
    .astype(int)
)

# order rows by the total number of predictors (p + q), largest first
datasets.sort(key=lambda key: int(added.loc[key].sum()), reverse=True)
count_fields = [
    'null_continuous_selected', 'null_categorical_selected', 'null_total_selected',
    'real_selected', 'real_categorical_selected',
]
stats = frame.groupby(['dataset', 'selector'])[count_fields].agg(['mean', 'std'])


def n_continuous(dataset):
    return int(added.loc[dataset, 'null_continuous_added'])


def n_categorical(dataset):
    return int(added.loc[dataset, 'null_categorical_added'])


def stat(dataset, selector, field, which):
    value = stats.loc[(dataset, selector), (field, which)]
    return 0.0 if pd.isna(value) else float(value)


def disp(selector):
    return selector_display.get(selector, selector)


shown_selectors = [disp(s) for s in selectors]
has_gb = any(s.startswith('GB') for s in selectors)

#----------------------------------------------------------------
# Layout: by_dataset
#----------------------------------------------------------------
def cell_by_dataset(dataset, selector):
    total = stat(dataset, selector, 'null_total_selected', 'mean')
    cont = stat(dataset, selector, 'null_continuous_selected', 'mean')
    if as_percent:
        p, q = n_continuous(dataset), n_categorical(dataset)
        total = 100 * total / (p + q)
        cont = 100 * cont / p if p else 0.0
        d = percent_decimals
    else:
        d = decimals
    return f'{total:.{d}f} ({cont:.{d}f})'


def render_by_dataset():
    quantity = 'percent of nulls selected' if as_percent else 'mean nulls selected'
    print(f'Synthetic null predictors selected by IPSS  '
          f'({quantity}, continuous in parentheses, over {n_draws} draws)')
    print()

    name_width = 22
    num_width = 6
    cell_width = 16
    header = (
        f'{"dataset":{name_width}s}{"n":>{num_width}s}{"p":>{num_width}s}{"q":>{num_width}s}'
        + ''.join(f'{name:>{cell_width}s}' for name in shown_selectors)
    )
    print(header)
    print('-' * len(header))
    for key in datasets:
        n = dataset_n.get(key, '')
        line = (
            f'{dataset_titles[key]:{name_width}s}{str(n):>{num_width}s}'
            f'{n_continuous(key):>{num_width}d}{n_categorical(key):>{num_width}d}'
        )
        for selector in selectors:
            line += f'{cell_by_dataset(key, selector):>{cell_width}s}'
        print(line)

    print()
    print('=' * 64)
    print()
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\small')
    print(r'\setlength{\tabcolsep}{6pt}')
    print(r'\renewcommand{\arraystretch}{1.05}')
    print(rf'\begin{{tabular}}{{lccc{"c" * len(selectors)}}}')
    print(r'\toprule')
    print(' & '.join([r'Dataset', r'$n$', r'$p$', r'$q$'] + shown_selectors) + r' \\')
    print(r'\midrule')
    for key in datasets:
        cells = [
            dataset_titles[key], str(dataset_n.get(key, '')),
            str(n_continuous(key)), str(n_categorical(key)),
        ]
        cells += [cell_by_dataset(key, selector) for selector in selectors]
        print(' & '.join(cells) + r' \\')
    print(r'\bottomrule')
    print(r'\end{tabular}')

    if as_percent:
        lead = (
            r'the percentage of the added null predictors selected by each base '
            rf'selector at target FDR $\alpha={target_fdr_label}$, averaged over '
            rf'${n_draws}$ draws, with the percentage of null continuous '
            r'predictors in parentheses'
        )
    else:
        lead = (
            r'the mean number of synthetic null predictors selected by each base '
            rf'selector at target FDR $\alpha={target_fdr_label}$ over ${n_draws}$ '
            r'draws, with the number that are continuous in parentheses'
        )
    caption = (
        r'\caption{\textit{Synthetic null predictors selected by IPSS}. For each '
        r'dataset ($n$ samples, $p$ continuous and $q$ categorical predictors), '
        + lead
        + r'. Each dataset is augmented with $p$ null continuous and $q$ null '
        r'categorical predictors, one per real predictor, formed as independent '
        r'row permutations that preserve the marginals.'
    )
    if not has_gb:
        caption += r' Gradient-boosted selectors also select no nulls (\cref{supsec:ml}).'
    caption += r'}'
    print(caption)
    print(rf'\label{{{latex_label}}}')
    print(r'\end{table}')


#----------------------------------------------------------------
# Layout: by_selector
#----------------------------------------------------------------
def render_by_selector():
    base_metrics = [
        ('null_continuous_selected', 'null cont.', 'Null cont.', n_continuous),
        ('null_categorical_selected', 'null cat.', 'Null cat.', n_categorical),
    ]
    if show_real:
        base_metrics.append(
            ('real_selected', 'real', 'Real',
             lambda d: n_continuous(d) + n_categorical(d))
        )
    if show_real_categorical:
        base_metrics.append(
            ('real_categorical_selected', 'real cat.', 'Real cat.', n_categorical)
        )

    suffix_plain = ' %' if as_percent else ''
    suffix_latex = r'\ (\%)' if as_percent else ''
    metrics = [
        (field, plain + suffix_plain, head + suffix_latex, denom)
        for field, plain, head, denom in base_metrics
    ]

    def value(dataset, selector, field, denom, with_sd, with_denominator):
        mean = stat(dataset, selector, field, 'mean')
        sd = stat(dataset, selector, field, 'std')
        if as_percent:
            base = denom(dataset)
            mean, sd = 100 * mean / base, 100 * sd / base
            text = f'{mean:.{percent_decimals}f}'
            return text + (f' ({sd:.{percent_decimals}f})' if with_sd else '')
        text = f'{mean:.{decimals}f}'
        if with_sd:
            text += f' ({sd:.{decimals}f})'
        if with_denominator:
            text += f' / {denom(dataset)}'
        return text

    quantity = 'percent of available' if as_percent else 'mean count; nulls as selected / added'
    print(f'Synthetic null predictors selected by IPSS  ({quantity}, over {n_draws} draws)')
    print()
    name_width, sel_width, col_width = 22, 14, 16
    header = f'{"dataset":{name_width}s}{"selector":{sel_width}s}' + ''.join(
        f'{label:>{col_width}s}' for _, label, _, _ in metrics
    )
    print(header)
    print('-' * len(header))
    for key in datasets:
        shown_title = dataset_titles[key]
        for selector in selectors:
            line = f'{shown_title:{name_width}s}{disp(selector):{sel_width}s}'
            shown_title = ''
            for field, _, _, denom in metrics:
                line += f'{value(key, selector, field, denom, plain_text_sd, True):>{col_width}s}'
            print(line)

    print()
    print('=' * 64)
    print()
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\small')
    print(r'\setlength{\tabcolsep}{6pt}')
    print(r'\renewcommand{\arraystretch}{1.05}')
    print(rf'\begin{{tabular}}{{ll{"c" * len(metrics)}}}')
    print(r'\toprule')
    print(' & '.join(['Dataset', 'Selector'] + [head for _, _, head, _ in metrics]) + r' \\')
    print(r'\midrule')
    for index, key in enumerate(datasets):
        if index > 0:
            print(r'\midrule')
        row_label = f'{dataset_titles[key]} (${n_continuous(key)}+{n_categorical(key)}$)'
        for selector in selectors:
            cells = [
                value(key, selector, field, denom, latex_sd, False)
                for field, _, _, denom in metrics
            ]
            print(f'{row_label} & {disp(selector)} & ' + ' & '.join(cells) + r' \\')
            row_label = ''
    print(r'\bottomrule')
    print(r'\end{tabular}')
    unit = 'Percentage of the added null predictors of each type' if as_percent \
        else 'Mean number of synthetic null predictors'
    print(
        rf'\caption{{\textit{{Synthetic null predictors selected by IPSS}}. {unit} '
        rf'selected by each base selector at target FDR $\alpha={target_fdr_label}$, '
        rf'averaged over ${n_draws}$ draws. Each dataset is augmented with one null '
        r'per real predictor (independent row permutations preserving the '
        r'marginals); the parenthetical is the number of null continuous $+$ null '
        r'categorical predictors added.}'
    )
    print(rf'\label{{{latex_label}}}')
    print(r'\end{table}')


if layout == 'by_dataset':
    render_by_dataset()
elif layout == 'by_selector':
    render_by_selector()
else:
    raise SystemExit(f"unknown layout {layout!r}; use 'by_dataset' or 'by_selector'")
