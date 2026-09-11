"""Shared settings and methods for the AUC feature-ranking study, used by
auc_ranking.py and jitter_strength_sensitivity.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from methods.jitter import jitterRF, jitterXGB, run_cforest, run_ufi


# ----------------------------------------------------------------
# Simulation settings
# ----------------------------------------------------------------
N = 200
P_CNTS = 25
P_BIN = 25
P_CORR = 25
N_TRUE_CNTS = 5
N_TRUE_BIN = 5
SNR = 1
N_TRIALS = 100
RANDOM_SEED = 123

RF_ARGS = {}
JITTER_STRENGTH = 0.0001

RF_METHOD_NAMES = (
    "RF",
    "RF-one-time",
    "RF-per-tree",
    "UFI",
    "CForest",
)
XGB_METHOD_NAMES = (
    "XGB-default",
    "XGB-stumps",
    "XGB-jitter-default",
    "XGB-jitter-stumps",
)

# Color-blind-friendly colors and line styles keep the curves distinguishable
# both on screen and when the figure is printed in grayscale.
ROC_STYLES = {
    "RF": {"color": "#0072B2", "linestyle": "-"},
    "RF-one-time": {"color": "#E69F00", "linestyle": "--"},
    "RF-per-tree": {"color": "#009E73", "linestyle": "-."},
    "UFI": {"color": "#D55E00", "linestyle": "-"},
    "CForest": {"color": "#CC79A7", "linestyle": ":"},
    "XGB-default": {"color": "#0072B2", "linestyle": "-"},
    "XGB-stumps": {"color": "#E69F00", "linestyle": "--"},
    "XGB-jitter-default": {"color": "#009E73", "linestyle": "-."},
    "XGB-jitter-stumps": {"color": "#D55E00", "linestyle": ":"},
}


def make_methods():
    return {
        "RF": lambda X, y: jitterRF(
            X, y, task="regression", jitter_method=None, **RF_ARGS
        ),
        "RF-one-time": lambda X, y: jitterRF(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=JITTER_STRENGTH,
            **RF_ARGS,
        ),
        "RF-per-tree": lambda X, y: jitterRF(
            X,
            y,
            task="regression",
            jitter_method="per_tree",
            jitter_strength=JITTER_STRENGTH,
            **RF_ARGS,
        ),
        "UFI": lambda X, y: run_ufi(X, y, task="regression", **RF_ARGS),
        "CForest": lambda X, y: run_cforest(X, y, task="regression"),
        "XGB-default": lambda X, y: jitterXGB(
            X,
            y,
            task="regression",
            jitter_method=None,
            max_depth=None,
        ),
        "XGB-stumps": lambda X, y: jitterXGB(
            X,
            y,
            task="regression",
            jitter_method=None,
            max_depth=1,
        ),
        "XGB-jitter-default": lambda X, y: jitterXGB(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=JITTER_STRENGTH,
            max_depth=None,
        ),
        "XGB-jitter-stumps": lambda X, y: jitterXGB(
            X,
            y,
            task="regression",
            jitter_method="one_time",
            jitter_strength=JITTER_STRENGTH,
            max_depth=1,
        ),
    }


def sample_true_features(trial):
    """Randomly choose the true continuous and binary features for a trial."""
    rng = np.random.default_rng(RANDOM_SEED + 2_000_000 + trial)
    true_cnts_idx = np.sort(
        rng.choice(P_CNTS, size=N_TRUE_CNTS, replace=False)
    )
    true_bin_idx = np.sort(
        rng.choice(P_BIN, size=N_TRUE_BIN, replace=False)
    )
    return true_cnts_idx, true_bin_idx
