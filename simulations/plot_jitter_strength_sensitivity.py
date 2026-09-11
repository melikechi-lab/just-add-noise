# Jitter-strength sensitivity figure, rebuilt from the published summary
"""Redraws the 2x2 feature-recovery AUC vs jitter-strength figure without rerunning
the experiment. The per-trial results were not saved, but the mean and standard
deviation for every (setting, method, delta) are in Table `tab:jitter_sensitivity`
of the paper and are hard-coded below. Uses save_sensitivity_figure from
jitter_strength_sensitivity.py, so styling matches the original exactly.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from jitter_strength_sensitivity import save_sensitivity_figure

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
# save_sensitivity_figure only writes files (Agg backend, no plt.show), so the
# figure is always written to output_path; show_fig then opens that file.
show_fig = True
output_path = Path(__file__).resolve().parent / 'jitter_strength_sensitivity.png'

# delta values, in the column order of tab:jitter_sensitivity
deltas = (1e-6, 1e-4, 1e-2, 1e-1)

# (dependence, signal_type, method): [(mean_auc, std_auc) per delta]
table = {
    ('independent', 'linear', 'RF-one-time'): [(0.945, 0.039), (0.944, 0.041), (0.944, 0.040), (0.944, 0.040)],
    ('independent', 'linear', 'XGB-one-time'): [(0.843, 0.066), (0.842, 0.062), (0.838, 0.064), (0.838, 0.064)],
    ('independent', 'nonlinear', 'RF-one-time'): [(0.891, 0.059), (0.891, 0.058), (0.890, 0.058), (0.890, 0.058)],
    ('independent', 'nonlinear', 'XGB-one-time'): [(0.789, 0.076), (0.777, 0.076), (0.774, 0.074), (0.774, 0.074)],
    ('correlated', 'linear', 'RF-one-time'): [(0.852, 0.067), (0.857, 0.067), (0.858, 0.067), (0.858, 0.067)],
    ('correlated', 'linear', 'XGB-one-time'): [(0.784, 0.085), (0.775, 0.081), (0.773, 0.081), (0.773, 0.081)],
    ('correlated', 'nonlinear', 'RF-one-time'): [(0.862, 0.062), (0.862, 0.063), (0.862, 0.064), (0.861, 0.063)],
    ('correlated', 'nonlinear', 'XGB-one-time'): [(0.765, 0.075), (0.767, 0.078), (0.768, 0.079), (0.768, 0.079)],
}

#----------------------------------------------------------------
# Build the summary frame and draw
#----------------------------------------------------------------
records = []
for (dependence, signal_type, method), entries in table.items():
    for delta, (mean_auc, std_auc) in zip(deltas, entries):
        records.append({
            'dependence': dependence,
            'signal_type': signal_type,
            'method': method,
            'epsilon': delta,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
        })

summary_df = pd.DataFrame(records)

save_sensitivity_figure(summary_df, output_path)
print(f'saved {output_path} (and .pdf)')

if show_fig:
    opener = {'darwin': 'open', 'win32': 'start'}.get(sys.platform, 'xdg-open')
    subprocess.run([opener, str(output_path)], check=False)
