# Split points
"""Figure 1 (fig:structural_asymmetry): a handful of samples from an independent
(X, Z) pair with X continuous and Z Bernoulli. The left panel shows how few splits
the binary Z admits; the right panel shows that adding uniform noise to Z gives it
as many candidate splits as X.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
save_fig = True

# figure params
dpi = 300
fontsize_axes = 20
fontsize_title = 24
linewidth = 1.5
point_size = 20

random_seed = 1
np.random.seed(random_seed)

n = 5              # small sample so the split points are countable (see caption)
epsilon = 0.1
output_path = Path(__file__).resolve().parent / 'figures' / 'split_points.png'

#----------------------------------------------------------------
# Helpers
#----------------------------------------------------------------
def split_points(values):
    sorted_values = np.sort(np.unique(values))
    return (sorted_values[:-1] + sorted_values[1:]) / 2


#----------------------------------------------------------------
# Generate samples
#----------------------------------------------------------------
X = np.random.normal(0, 1, n)
Z = np.random.binomial(1, 0.5, n)
Z_perturbed = Z + np.random.uniform(-epsilon, epsilon, n)

x_split_points = split_points(X)
z_perturbed_split_points = split_points(Z_perturbed)

#----------------------------------------------------------------
# Plot
#----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16,6), sharex=True, sharey=True)

axes[0].scatter(X, Z, s=point_size)
for split_point in x_split_points:
    axes[0].axvline(split_point, color='gray', linewidth=linewidth, alpha=0.75, linestyle='--')
axes[0].axhline(0.5, color='tab:red', linewidth=linewidth, alpha=0.75, linestyle='--')
# axes[0].set_title('Binary sample', fontsize=fontsize_title)
axes[0].set_xlabel('$X\\sim\\mathcal{N}(0,1)$', fontsize=fontsize_axes)
axes[0].set_ylabel(f'$Z\\sim$ Bernoulli$(1/2)$', fontsize=fontsize_axes)
axes[0].set_yticks([0, 0.5, 1])

axes[1].scatter(X, Z_perturbed, s=point_size)
for split_point in x_split_points:
    axes[1].axvline(split_point, color='gray', linewidth=linewidth, alpha=0.75, linestyle='--')
for split_point in z_perturbed_split_points:
    axes[1].axhline(split_point, color='tab:red', linewidth=linewidth, alpha=0.75, linestyle='--')
# axes[1].set_title('Binary sample with uniform noise', fontsize=fontsize_title)
axes[1].set_xlabel('$X\\sim\\mathcal{N}(0,1)$', fontsize=fontsize_axes)

plt.tight_layout()
if save_fig:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f'saved {output_path}')
plt.show()
