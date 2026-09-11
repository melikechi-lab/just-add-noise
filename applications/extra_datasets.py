# Generic OpenML/PMLB dataset loader
"""Loads a dataset straight from OpenML or PMLB and encodes it the same way as
clean_selected_datasets.load_and_clean_dataset: continuous/categorical split by the
<=20-distinct-value rule (methods.dataset_types.classify_column), median imputation for
continuous columns, missing categorical values as their own level, integer coding.

Used for the extra datasets (German Credit, QSAR Biodegradation, Hypothyroid) alongside
the six in clean_selected_datasets.py.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openml
import pandas as pd
import pmlb

from clean_selected_datasets import DEFAULT_PMLB_CACHE
from methods.dataset_types import classify_column

# the 41 QSAR biodegradation molecular descriptors, in the fixed UCI column order
# (the OpenML 1494 upload names them V1..V41); source: Mansouri et al. (2013).
QSAR_BIODEG_DESCRIPTORS = [
    'SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%',
    'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L', 'F03[C-O]',
    'Me', 'Mi', 'nN-N', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 'nCIR', 'B01[C-Br]',
    'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L',
    'nCrt', 'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN',
    'SM6_B(m)', 'nArCOOR', 'nX',
]


def load_extra_dataset(spec):
    """Generic OpenML/PMLB loader. Returns
    (X_df, y_series, cat_idx, task, title, raw_missing, notes)."""
    if spec['source'] == 'openml':
        data = openml.datasets.get_dataset(
            spec['id'], download_data=True,
            download_qualities=False, download_features_meta_data=False,
        )
        target = spec['target'] or data.default_target_attribute
        X, y = data.get_data(target=target)[:2]
    else:
        frame = pmlb.fetch_data(spec['id'], local_cache_dir=str(DEFAULT_PMLB_CACHE))
        y = frame['target']
        X = frame.drop(columns=['target'])

    X = X.drop(columns=spec.get('drop', []), errors='ignore')
    rename = spec.get('rename')
    if isinstance(rename, (list, tuple)):
        if len(rename) != X.shape[1]:
            raise ValueError(
                f"{spec['id']}: rename list has {len(rename)} names for {X.shape[1]} columns"
            )
        X.columns = list(rename)
    elif isinstance(rename, dict):
        X = X.rename(columns=rename)
    keep = y.notna().to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

    continuous, categorical, raw_missing = [], [], {}
    for column in X.columns:
        raw_missing[column] = int(X[column].isna().sum())
        kind = classify_column(X[column])[0]
        if kind == 'continuous':
            continuous.append(column)
        elif kind == 'categorical':
            categorical.append(column)
        # 'constant' columns are dropped

    encoded = pd.DataFrame(index=X.index)
    for column in continuous:
        values = pd.to_numeric(X[column], errors='coerce')
        encoded[column] = values.fillna(values.median())
    for column in categorical:
        values = X[column].astype('string').fillna('__MISSING__')
        levels = sorted(values.unique())
        encoded[column] = values.map({lvl: i for i, lvl in enumerate(levels)}).astype(int)

    if spec['task'] == 'classification':
        y_series = pd.Series(pd.factorize(y)[0], name='target')
    else:
        y_series = pd.to_numeric(y, errors='coerce').rename('target')

    cat_idx = list(range(len(continuous), len(continuous) + len(categorical)))
    notes = [f"{spec['source']}:{spec['id']}, generic recipe (<=20-distinct type rule, "
             f"median-imputed continuous, missing categorical as own level)"]
    return encoded, y_series, cat_idx, spec['task'], spec['title'], raw_missing, notes
