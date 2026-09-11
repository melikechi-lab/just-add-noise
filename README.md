# Just add noise – paper repository

This repository accompanies the paper:

**Just add noise: Debiasing tree-based variable importance in mixed data**

The method: add a small uniform jitter to each categorical predictor before fitting a
tree ensemble. This repo reproduces every figure and table in the paper.

---

## Installation

```bash
python -m pip install -r requirements.txt
```

The CForest baseline (Table 1, Table S1, Fig. 4) also needs R with `partykit`,
`libcoin`, and `mvtnorm` installed; every script skips it gracefully if R is absent.

---

## Quickstart

No data needed:
```bash
cd simulations
python split_points.py          # Figure 1
```

Run scripts from within their own directory. A `*_study.py` / `run_*.py` script writes
results next to itself; the paired `plot_*.py` / `report_*.py` script makes the figure
or table.

---

## What produces what

| Script | Figure / table |
|---|---|
| `simulations/split_points.py` | Fig. 1 |
| `simulations/nonnull_importance.py` | Fig. 2 |
| `simulations/auc_ranking.py` | Table 1, Fig. S1 |
| `simulations/ipss_simulation.py` → `plot_ipss_simulation.py` | Fig. 3 |
| `simulations/runtime.py` | Table S1 |
| `simulations/jitter_strength_sensitivity.py` → `plot_jitter_strength_sensitivity.py` | Table S3, Fig. S4 |
| `applications/bladder_18m_survival_mirna_ordinal_stage.py` → `plot_rank_boxplots.py` | Fig. 4, Table S2 |
| `applications/ipss_null_spikein_study.py` → `report_ipss_null_spikein_table.py` | Table 2 |
| `applications/ipss_selection_study.py` → `plot_ipss_selection_cardinality.py` | Fig. S2 |
| `applications/rank_null_spikein_study.py` → `plot_rank_null_spikein_heatmap.py` | Fig. S3 |
| `applications/prediction/run_prediction.py` → `plot_error_ratio.py` | Fig. S5 |

`methods/` holds the shared code (`jitter.py`, `ufi.py`, `data_generation.py`,
`dataset_types.py`) that these scripts import.

---

## Datasets

No data is stored here. The machine-learning and prediction datasets download
automatically from OpenML and PMLB. The bladder cancer cohort needs a manual download
from LinkedOmics — see the comment at the top of
`applications/bladder_18m_survival_mirna_ordinal_stage.py`.

---

## Contact

Omar Melikechi, omar.melikechi@gmail.com
