# Find PMLB datasets with at least one obviously continuous and one obviously categorical predictor
"""
See dataset_types.classify_column for how each predictor is classified.
A dataset qualifies if it has at least one continuous and at least one categorical predictor.
(An earlier version of this screen also accepted all-categorical datasets whose predictors
varied in cardinality, on the reasoning that the split-point-count bias jitter corrects isn't
specific to the categorical/continuous cutoff. That widened pool turned out to concentrate
jitter's worst error-ratio outliers on all-categorical, uniformly-low-cardinality, often
deterministic-rule datasets like the MONK problems -- plausibly because there's no real
split-count imbalance for jitter to correct there, so it just adds noise. Reverted to
requiring a genuine continuous predictor to avoid that failure mode.)
Dataset names are read from all_summary_stats_pmlb.tsv (downloaded from the pmlb github) rather
than pmlb.dataset_names, since the installed pmlb package's built-in list is version-pinned and
only covers 284 of PMLB's current 450 datasets.
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pmlb

from methods.dataset_types import classify_column

print()

#----------------------------------------------------------------
# Settings
#----------------------------------------------------------------
min_predictors = 2
max_predictors = 5000
max_workers = 8
script_dir = Path(__file__).resolve().parent
# regenerates dataset_lists/clean_datasets_pmlb_final.csv (which is checked in);
# all_summary_stats_pmlb.tsv is downloaded from the PMLB GitHub repo
summary_stats_tsv = script_dir / 'dataset_lists' / 'all_summary_stats_pmlb.tsv'
output_csv = script_dir / 'dataset_lists' / 'clean_datasets_pmlb.csv'
pmlb_cache_dir = script_dir / 'pmlb_cache'  # shared with run_prediction.py, avoids re-downloading

fetch_retries = 3
fetch_retry_delay = 5

#----------------------------------------------------------------
# Per-dataset check using actual downloaded data
#----------------------------------------------------------------
def check_dataset(name):
    # pmlb downloads occasionally hit transient connection drops (e.g. RemoteDisconnected
    # from github closing the connection mid-request); retry a few times before giving up
    df = None
    for attempt in range(1, fetch_retries + 1):
        try:
            df = pmlb.fetch_data(name, local_cache_dir=pmlb_cache_dir)
            break
        except Exception as e:
            if attempt == fetch_retries:
                print(f'  skipping {name}: {e}')
                return None
            time.sleep(fetch_retry_delay)

    predictor_names = [c for c in df.columns if c != 'target']
    if min_predictors is not None and len(predictor_names) < min_predictors:
        return None
    if max_predictors is not None and len(predictor_names) > max_predictors:
        return None

    continuous_vars = []
    categorical_vars = []
    distinct_counts = []
    for name_ in predictor_names:
        kind, n_distinct = classify_column(df[name_])
        if kind == 'constant':
            continue
        distinct_counts.append(n_distinct)
        if kind == 'continuous':
            continuous_vars.append(name_)
        else:
            categorical_vars.append(name_)

    if not continuous_vars or not categorical_vars:
        return None

    return {
        'name': name,
        'n_rows': len(df),
        'n_predictors': len(predictor_names),
        'n_continuous': len(continuous_vars),
        'n_categorical': len(categorical_vars),
        'min_distinct': min(distinct_counts),
        'max_distinct': max(distinct_counts),
        'continuous_vars': ';'.join(continuous_vars),
        'categorical_vars': ';'.join(categorical_vars),
    }

names = pd.read_csv(summary_stats_tsv, sep='\t')['dataset'].tolist()
print(f'{len(names)} datasets in {summary_stats_tsv}')

rows = []
n_names = len(names)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(check_dataset, name) for name in names]
    for i, future in enumerate(as_completed(futures), start=1):
        if i % 20 == 0:
            print(f'checked {i}/{n_names} datasets...')
        result = future.result()
        if result is not None:
            rows.append(result)

#----------------------------------------------------------------
# Write output
#----------------------------------------------------------------
fieldnames = ['name', 'n_rows', 'n_predictors', 'n_continuous', 'n_categorical', 'min_distinct', 'max_distinct', 'continuous_vars', 'categorical_vars']
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'found {len(rows)} qualifying datasets, written to {output_csv}')
