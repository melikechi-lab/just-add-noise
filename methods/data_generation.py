# Simulation data-generating processes
"""Mixed continuous/binary designs used by the simulation studies.

Two designs are provided:

* ``cnts_bin_indep`` -- all predictors mutually independent.
* ``cnts_bin_corr``  -- the first ``p_corr`` continuous/binary pairs are correlated.

Both return ``(X, y)`` with ``X = [X_cnts | X_bin]`` (continuous columns first) and a
continuous response ``y = f(X_s) + noise`` at the requested signal-to-noise ratio. The
signal ``f`` is linear or a standardized Gaussian bump ('nonlinear'); see the manuscript
supplement, 'Data-generating process'.
"""

import numpy as np


def _make_signal(x_cnts, x_bin, true_cnts_idx, true_bin_idx, signal_type, rng):
    """Build f(X_s) from the true continuous / binary features.

    linear     : f = sum_j sign * X_j        + sum_j 2 * sign * Z_j
    nonlinear  : f = sum_j sign * g(X_j)      + sum_j 2 * sign * Z_j,
                 with g(x) = exp(-x^2 / 2) standardized to mean 0, variance 1
                 under X ~ N(0, 1).
    """
    n, p_cnts = x_cnts.shape
    p_bin = x_bin.shape[1]
    signal = np.zeros(n)

    for idx in true_cnts_idx:
        if idx >= p_cnts:
            continue
        sign = rng.choice([-1, 1])

        if signal_type == 'linear':
            signal += sign * x_cnts[:, idx]

        elif signal_type == 'nonlinear':
            g = np.exp(-0.5 * x_cnts[:, idx] ** 2)
            # theoretical mean and SD of g when X ~ N(0, 1)
            mean_g = 1 / np.sqrt(2)
            sd_g = np.sqrt(1 / np.sqrt(3) - 1 / 2)
            g = (g - mean_g) / sd_g
            signal += sign * g

    # binary Z features contribute linearly (they are 0/1) in both settings
    for idx in true_bin_idx:
        if idx >= p_bin:
            continue
        sign = rng.choice([-1, 1])
        signal += 2 * sign * x_bin[:, idx]

    return signal


def _add_noise(signal, snr, rng):
    """Add Gaussian noise to achieve the desired signal-to-noise ratio."""
    n = signal.shape[0]
    signal_var = np.var(signal)
    if signal_var == 0:
        return rng.normal(0, 1, size=n)
    sigma_noise = np.sqrt(signal_var / snr)
    return signal + rng.normal(0, sigma_noise, size=n)


def _as_idx(idx):
    if idx is None:
        return np.array([], dtype=int)
    return np.array(idx, dtype=int)


def cnts_bin_corr(n, p_cnts, p_bin,
                  p_corr,                # number of (X_j, Z_j) pairs that are correlated
                  true_cnts_idx=None,    # indices of the true continuous predictors
                  true_bin_idx=None,     # indices of the true binary predictors
                  target_rho=0.8,        # target correlation within a correlated pair
                  snr=1.0,
                  signal_type='linear',  # 'linear' | 'nonlinear'
                  seed=None):
    """X and Z where the first ``p_corr`` pairs (X_j, Z_j) are correlated."""

    rng = np.random.default_rng(seed)

    true_cnts_idx = _as_idx(true_cnts_idx)
    true_bin_idx = _as_idx(true_bin_idx)

    # mu_pos / mu_neg chosen to hit the target correlation
    if target_rho > 0 and target_rho < 1:
        delta = (2 * target_rho) / np.sqrt(1 - target_rho ** 2)
        mu_pos, mu_neg = delta / 2, -delta / 2
    else:
        mu_pos, mu_neg = 0, 0

    # binary features
    x_bin = rng.binomial(1, 0.5, size=(n, p_bin))

    # continuous features with the specified correlation structure
    x_cnts = np.zeros((n, p_cnts))

    limit = min(p_corr, p_cnts, p_bin)
    if limit > 0:
        eps_corr = rng.normal(0, 1, size=(n, limit))
        z_subset = x_bin[:, :limit]
        x_cnts[:, :limit] = z_subset * (mu_pos + eps_corr) + (1 - z_subset) * (mu_neg + eps_corr)

    if p_cnts > limit:
        x_cnts[:, limit:] = rng.normal(0, 1, size=(n, p_cnts - limit))

    X = np.hstack([x_cnts, x_bin])

    signal = _make_signal(x_cnts, x_bin, true_cnts_idx, true_bin_idx, signal_type, rng)
    y = _add_noise(signal, snr, rng)

    return X, y


def cnts_bin_indep(n, p_cnts, p_bin,
                   true_cnts_idx=None,    # indices of the true continuous predictors
                   true_bin_idx=None,     # indices of the true binary predictors
                   snr=1.0,
                   signal_type='linear',  # 'linear' | 'nonlinear'
                   seed=None):
    """X and Z all mutually independent (no correlation structure)."""

    rng = np.random.default_rng(seed)

    true_cnts_idx = _as_idx(true_cnts_idx)
    true_bin_idx = _as_idx(true_bin_idx)

    x_bin = rng.binomial(1, 0.5, size=(n, p_bin))
    x_cnts = rng.normal(0, 1, size=(n, p_cnts))

    X = np.hstack([x_cnts, x_bin])

    signal = _make_signal(x_cnts, x_bin, true_cnts_idx, true_bin_idx, signal_type, rng)
    y = _add_noise(signal, snr, rng)

    return X, y
