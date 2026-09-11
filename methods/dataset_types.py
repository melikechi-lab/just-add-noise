# Categorical vs. continuous predictor classification
"""
A predictor is classified from its actual downloaded values, not OpenML's declared type:
  - numeric column, >max_categorical_distinct distinct non-missing values -> continuous
  - any column (numeric or string), 2..max_categorical_distinct distinct non-missing values -> categorical
  - fewer than 2 distinct non-missing values -> constant (not usable as a predictor)
String columns are normalized (stripped, casefolded) before counting distinct values so that
case variants of the same label (e.g. 'Male' / 'male') don't inflate the count.
"""
import pandas as pd

max_categorical_distinct = 20

def classify_column(col):
    non_null = col.dropna()
    if pd.api.types.is_numeric_dtype(col):
        n_distinct = non_null.nunique()
        if n_distinct < 2:
            return 'constant', n_distinct
        elif n_distinct <= max_categorical_distinct:
            return 'categorical', n_distinct
        else:
            return 'continuous', n_distinct
    else:
        normalized = non_null.astype(str).str.strip().str.casefold()
        n_distinct = normalized.nunique()
        if n_distinct < 2:
            return 'constant', n_distinct
        else:
            return 'categorical', n_distinct
