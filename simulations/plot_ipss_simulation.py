# IPSS simulation figure (Figure 3)
"""One 2-by-4 figure per feature category (all, cont, bin): TPR on the top row, FDR on
the bottom row, one column per simulation setting. Reads the pickles written by
ipss_simulation.py (run it for both signal types first)."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_fig = True
plot_fig = False

# methods to draw, in this order (also the legend order); keys must match the pickles
methods_to_plot = ['RF', 'RF-jitter', 'UFI', 'GB', 'GB-jitter']
# methods_to_plot = ['GB', 'GB-jitter']

# how each method is labelled in the legend
method_labels = {
    'RF': 'RF',
    'RF-jitter': 'RF-one-time',
    'UFI': 'UFI',
    'GB': 'XGB',
    'GB-jitter': 'XGB-one-time',
}

categories = ['all'] #['all', 'cont', 'bin']

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
output_dir = script_dir / 'results' / 'figures'

dpi = 300

# (results folder, panel title)
scenarios = [
    ('independent_linear', 'Independent\nlinear'),
    ('correlated_linear', 'Correlated\nlinear'),
    ('independent_nonlinear', 'Independent\nnonlinear'),
    ('correlated_nonlinear', 'Correlated\nnonlinear'),
]

# a base method and its '-one-time' variant share a color (jitter solid, base dashed)
method_styles = {
    'GB': {'color': 'dodgerblue', 'linestyle': ':', 'linewidth': 2},
    'GB-jitter': {'color': 'dodgerblue', 'linestyle': '-', 'linewidth': 2},
    'RF': {'color': 'limegreen', 'linestyle': ':', 'linewidth': 2},
    'RF-jitter': {'color': 'limegreen', 'linestyle': '-', 'linewidth': 2},
    'UFI': {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
}
default_style = {'color': '#666666', 'linestyle': '-', 'linewidth': 2.5}

plt.rcParams.update({
    'figure.dpi': 130,
    'savefig.dpi': dpi,
    'savefig.facecolor': 'white',
    'font.size': 11.5,
    'axes.labelsize': 13.5,
    'axes.titlesize': 14.0,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.9,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'legend.fontsize': 9.0,
    'legend.frameon': True,
    'legend.framealpha': 0.96,
    'legend.edgecolor': '#D0D0D0',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
def load(folder, name):
    with open(results_dir / folder / name, 'rb') as f:
        return pickle.load(f)

def true_count(true_features, category):
    keys = {'cont': ['continuous'], 'bin': ['binary']}.get(category, ['continuous', 'binary'])
    return np.array([sum(len(t[k]) for k in keys) for t in true_features], dtype=float)

#----------------------------------------------------------------
# Plotting
#----------------------------------------------------------------
def style_axis(ax, show_xlabels):
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_axisbelow(True)
    ax.grid(True, color='#E2E2E2', linewidth=0.75)
    ax.tick_params(direction='out', labelbottom=show_xlabels)
    for spine in ax.spines.values():
        spine.set_color('#4A4A4A')

def make_figure(category):
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.3), sharex=True, sharey='row')

    for col, (folder, title) in enumerate(scenarios):
        alphas = np.asarray(load(folder, 'alphas.pkl'), dtype=float)
        denom = true_count(load(folder, 'true_features_by_trial.pkl'), category)[:, None]
        tp = load(folder, f'TP_{category}.pkl')
        fp = load(folder, f'FP_{category}.pkl')
        ns = load(folder, f'NS_{category}.pkl')

        tpr_ax, fdr_ax = axes[0, col], axes[1, col]
        style_axis(tpr_ax, show_xlabels=False)
        style_axis(fdr_ax, show_xlabels=True)
        tpr_ax.set_title(title, pad=8)

        for method in methods_to_plot:
            style = method_styles.get(method, default_style)
            tpr = np.asarray(tp[method], dtype=float) / denom
            fp_m = np.asarray(fp[method], dtype=float)
            ns_m = np.asarray(ns[method], dtype=float)
            fdr = np.divide(fp_m, ns_m, out=np.zeros_like(fp_m), where=ns_m > 0)  # FDR = 0 when nothing selected
            for ax, values in ((tpr_ax, tpr), (fdr_ax, fdr)):
                ax.plot(alphas, values.mean(axis=0), color=style['color'], linestyle=style['linestyle'],
                    linewidth=style['linewidth'], solid_capstyle='round', dash_capstyle='round', zorder=3)

        fdr_ax.plot(alphas, alphas, color='#222222', linestyle='--', linewidth=1.8, zorder=4)

    axes[0, 0].set_ylabel('TPR', fontsize=18)
    axes[1, 0].set_ylabel('FDR', fontsize=18)
    fig.supxlabel('Target FDR', fontsize=16, y=0.03)

    handles = [
        Line2D([0], [0], label=method_labels.get(method, method),
            color=method_styles.get(method, default_style)['color'],
            linestyle=method_styles.get(method, default_style)['linestyle'],
            linewidth=method_styles.get(method, default_style)['linewidth'])
        for method in methods_to_plot
    ]
    axes[0, -1].legend(handles=handles, loc='upper left', borderpad=0.6, handlelength=2.2, labelspacing=0.4)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.115, top=0.94, wspace=0.10, hspace=0.11)

    if save_fig:
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(output_dir / f'ipss_{category}_features.{ext}', bbox_inches='tight', pad_inches=0.08)

    if plot_fig:
        plt.show()
    plt.close(fig)

#----------------------------------------------------------------
# Run
#----------------------------------------------------------------
for category in categories:
    make_figure(category)

if save_fig:
    print(f'saved figures to {output_dir}')
else:
    print('save_fig is False; no files written')
