#!/usr/bin/env python3
"""Shared utilities for the real-data mixed-predictor applications.

The dataset-specific scripts are responsible for selecting the outcome and
clinical variables.  This module provides the common, auditable mechanics:

1. read the feature-by-sample text files and transpose them;
2. retain an omics panel that is complete across the source overlap;
3. optionally remove almost perfectly correlated continuous features;
4. save the assembled mixed dataset with readable categorical labels; and
5. run RF, jittered RF, XGBoost, jittered XGBoost, UFI, and CForest on the
   full dataset and save both raw importance scores and ranks.

No IPSS code is imported or run here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

NA_VALUES = [
    "NA",
    "NaN",
    "nan",
    "None",
    "[Not Available]",
    "[Not Evaluated]",
    "[Unknown]",
    "[Not Applicable]",
    "[not available]",
    "[not evaluated]",
    "[unknown]",
    "[not applicable]",
    "",
]

METHODS = (
    "rf",
    "rf_one_time",
    "rf_per_tree",
    "xgb",
    "xgb_one_time",
    "ufi",
    "cforest",
)


def read_feature_by_sample(path: Path) -> pd.DataFrame:
    """Read a tab-separated feature-by-sample table as sample-by-feature."""
    frame = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        na_values=NA_VALUES,
        keep_default_na=True,
    ).T
    frame.index = frame.index.astype(str).str.strip()
    frame.index.name = "SAMPLE_ID"
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicates[:5]}")
    return frame


def read_pooled_kidney_clinical(data_dir: Path) -> pd.DataFrame:
    """Pool the KIRC, KIPR, and KICH clinical tables by patient ID."""
    parts = [
        pd.read_csv(
            data_dir / filename,
            sep="\t",
            index_col=0,
            na_values=NA_VALUES,
            keep_default_na=True,
        )
        for filename in ("KIRC.txt", "KIPR.txt", "KICH.txt")
    ]
    clinical = pd.concat(parts, axis=1).T
    clinical.index = clinical.index.astype(str).str.strip()
    clinical.index.name = "SAMPLE_ID"
    if clinical.index.has_duplicates:
        raise ValueError("The pooled kidney clinical data contain duplicate sample IDs")
    return clinical


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize a clinical string column without converting missing values."""
    return series.astype("string").str.strip().str.lower()


def map_categories_strict(
    series: pd.Series,
    mapping: dict[str, str],
    label: str,
) -> pd.Series:
    """Normalize and map categories, rejecting any unexpected nonmissing label."""
    normalized = normalize_text(series)
    observed = set(normalized.dropna().unique().tolist())
    unexpected = sorted(observed - set(mapping))
    if unexpected:
        raise ValueError(f"Unexpected values in {label}: {unexpected}")
    return normalized.map(mapping).astype("string")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def ordered_overlap(left: pd.Index, right: pd.Index) -> pd.Index:
    """Return shared sample IDs in the order of the left-hand table."""
    return left[left.isin(right)]


def make_fixed_horizon_mortality(
    overall_survival: pd.Series,
    status: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """Construct a censoring-aware binary mortality endpoint.

    A death observed on or before ``horizon_days`` is coded as 1. A patient
    observed for at least ``horizon_days`` is coded as 0, whether the patient
    was still alive then or died later. Patients alive/censored before the
    horizon have an unknown endpoint and remain missing for complete-case
    exclusion by the dataset-specific application.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    survival_days = pd.to_numeric(overall_survival, errors="coerce")
    event = pd.to_numeric(status, errors="coerce")
    observed_events = set(event.dropna().unique().tolist())
    if not observed_events <= {0, 1}:
        raise ValueError(f"Unexpected status values: {sorted(observed_events)}")
    if (survival_days.dropna() < 0).any():
        raise ValueError("overall_survival contains negative follow-up times")

    outcome = pd.Series(pd.NA, index=survival_days.index, dtype="Int64")
    known_alive_at_horizon = survival_days.ge(horizon_days)
    death_by_horizon = event.eq(1) & survival_days.le(horizon_days)
    outcome.loc[known_alive_at_horizon] = 0
    outcome.loc[death_by_horizon] = 1
    return outcome


def _drop_almost_perfect_correlations(
    frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Greedily keep the first feature from each |correlation| > threshold pair."""
    if not 0 <= threshold <= 1:
        raise ValueError("correlation_threshold must be between 0 and 1")
    if frame.shape[1] < 2 or threshold == 1:
        return frame, []

    correlation = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    drop_positions: set[int] = set()
    for i in range(correlation.shape[0]):
        later = np.flatnonzero(
            np.abs(correlation[i, i + 1 :]) > threshold
        ) + i + 1
        drop_positions.update(int(position) for position in later)

    dropped = [frame.columns[position] for position in sorted(drop_positions)]
    kept_positions = [
        position
        for position in range(frame.shape[1])
        if position not in drop_positions
    ]
    return frame.iloc[:, kept_positions], dropped


def prepare_continuous_omics(
    omics: pd.DataFrame,
    panel_ids: Iterable[str],
    analysis_ids: Iterable[str],
    prefix: str,
    correlation_threshold: float = 0.999,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create a complete, nonconstant, optionally correlation-pruned omics panel.

    Completeness is assessed on ``panel_ids`` rather than only the final outcome
    complete cases.  This prevents cohort- or outcome-dependent missingness from
    deciding which molecular variables enter the analysis.
    """
    panel_ids = pd.Index(panel_ids)
    analysis_ids = pd.Index(analysis_ids)
    numeric = omics.apply(pd.to_numeric, errors="coerce")
    panel = numeric.loc[panel_ids]

    complete_columns = panel.columns[panel.notna().all(axis=0)]
    panel = panel.loc[:, complete_columns]
    analysis = numeric.loc[analysis_ids, complete_columns]

    nonconstant = analysis.columns[analysis.nunique(dropna=True) > 1]
    analysis = analysis.loc[:, nonconstant]
    before_correlation = analysis.shape[1]
    analysis, dropped_correlated = _drop_almost_perfect_correlations(
        analysis,
        correlation_threshold,
    )

    if analysis.isna().any().any():
        raise ValueError("The selected omics panel unexpectedly contains missing values")

    analysis.columns = [f"{prefix}::{name}" for name in analysis.columns]
    return analysis.astype(float), {
        "raw_features": int(omics.shape[1]),
        "complete_features": int(len(complete_columns)),
        "nonconstant_features": int(before_correlation),
        "correlation_pruned": int(len(dropped_correlated)),
        "retained_features": int(analysis.shape[1]),
    }


@dataclass
class MixedDataset:
    name: str
    task: str
    outcome_name: str
    outcome: pd.Series
    continuous: pd.DataFrame
    categorical: pd.DataFrame
    panel_report: dict[str, int]

    def validate(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        expected = self.outcome.index
        if not expected.equals(self.continuous.index):
            raise ValueError("Outcome and continuous predictors have different sample orders")
        if not expected.equals(self.categorical.index):
            raise ValueError("Outcome and categorical predictors have different sample orders")
        if self.outcome.isna().any():
            raise ValueError("Outcome contains missing values")
        if self.continuous.isna().any().any():
            raise ValueError("Continuous predictors contain missing values")
        if self.categorical.isna().any().any():
            raise ValueError("Categorical predictors contain missing values")
        if self.categorical.shape[1] < 1:
            raise ValueError("At least one original categorical predictor is required")
        if self.continuous.columns.intersection(self.categorical.columns).size:
            raise ValueError("Continuous and categorical predictor names overlap")

    @property
    def feature_names(self) -> list[str]:
        return self.continuous.columns.tolist() + self.categorical.columns.tolist()

    @property
    def feature_types(self) -> list[str]:
        return ["continuous"] * self.continuous.shape[1] + [
            "categorical"
        ] * self.categorical.shape[1]

    def assembled_frame(self) -> pd.DataFrame:
        self.validate()
        outcome = self.outcome.rename(self.outcome_name)
        return pd.concat([outcome, self.continuous, self.categorical], axis=1)

    def print_summary(self) -> None:
        self.validate()
        print(f"\n{self.name}")
        print("-" * len(self.name))
        print(
            f"n={len(self.outcome)}, continuous={self.continuous.shape[1]}, "
            f"categorical={self.categorical.shape[1]}, task={self.task}"
        )
        print(f"outcome={self.outcome_name}")
        if self.task == "classification":
            print(f"outcome counts: {self.outcome.value_counts().sort_index().to_dict()}")
        else:
            print("outcome summary:")
            print(self.outcome.astype(float).describe().round(4).to_string())
        print("categorical counts:")
        for column in self.categorical:
            counts = self.categorical[column].value_counts().to_dict()
            print(f"  {column}: {counts}")
        print(f"omics panel: {self.panel_report}")


def _encode_dataset(dataset: MixedDataset) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Encode readable categorical labels using the importance.py convention."""
    continuous = dataset.continuous.to_numpy(dtype=float)
    categorical_columns = []
    for column in dataset.categorical:
        values = dataset.categorical[column].astype(str)
        categories = sorted(values.unique().tolist())
        codes = pd.Categorical(values, categories=categories).codes
        if (codes < 0).any():
            raise ValueError(f"Failed to encode categorical variable {column}")
        categorical_columns.append(codes.astype(float))

    categorical = np.column_stack(categorical_columns)
    X = np.column_stack([continuous, categorical])
    if dataset.task == "classification":
        y = dataset.outcome.to_numpy(dtype=int)
    else:
        y = dataset.outcome.to_numpy(dtype=float)
    first_categorical = dataset.continuous.shape[1]
    cat_idx = list(range(first_categorical, X.shape[1]))
    return X, y, cat_idx


def _descending_min_ranks(values: np.ndarray) -> np.ndarray:
    """Rank largest values first, assigning the minimum rank to ties."""
    return (
        pd.Series(values)
        .rank(method="min", ascending=False)
        .astype(int)
        .to_numpy()
    )


def run_importance_methods(
    dataset: MixedDataset,
    n_estimators: int,
    max_depth: int | None,
    jitter_strength: float,
    random_seed: int,
    n_jobs: int,
    methods: Sequence[str] = METHODS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the selected full-data importance methods (without IPSS).

    A requested method that cannot run in this environment (currently only
    ``cforest`` without rpy2/R) is skipped with a printed notice rather than
    raising; the returned frame and ``runtimes`` only cover the methods that
    actually ran.
    """
    from methods.jitter import jitterRF, jitterXGB, run_cforest, run_ufi, unavailable_methods

    methods = tuple(dict.fromkeys(methods))
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown importance methods: {unknown}")
    if not methods:
        raise ValueError("At least one importance method must be selected")

    unavailable = set(unavailable_methods())
    for method in methods:
        if method in unavailable:
            print(f"Skipping {method}: not available in this environment (rpy2/R not installed).", flush=True)
    methods = tuple(method for method in methods if method not in unavailable)
    if not methods:
        raise ValueError("None of the requested importance methods are available in this environment")

    X, y, cat_idx = _encode_dataset(dataset)
    # All observations are used for fitting because this experiment compares
    # full-data importance rankings rather than held-out predictive error.
    rf_args = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_jobs": n_jobs,
    }

    np.random.seed(random_seed)
    functions = {
        "rf": lambda: jitterRF(
            X,
            y,
            task=dataset.task,
            jitter_method=None,
            cat_idx=cat_idx,
            **rf_args,
        ),
        "rf_one_time": lambda: jitterRF(
            X,
            y,
            task=dataset.task,
            jitter_method="one_time",
            jitter_strength=jitter_strength,
            cat_idx=cat_idx,
            **rf_args,
        ),
        "rf_per_tree": lambda: jitterRF(
            X,
            y,
            task=dataset.task,
            jitter_method="per_tree",
            jitter_strength=jitter_strength,
            cat_idx=cat_idx,
            **rf_args,
        ),
        "xgb": lambda: jitterXGB(
            X,
            y,
            task=dataset.task,
            jitter_method=None,
            cat_idx=cat_idx,
            max_depth=max_depth,
        ),
        "xgb_one_time": lambda: jitterXGB(
            X,
            y,
            task=dataset.task,
            jitter_method="one_time",
            jitter_strength=jitter_strength,
            cat_idx=cat_idx,
            max_depth=max_depth,
        ),
        "ufi": lambda: run_ufi(
            X,
            y,
            task=dataset.task,
            cat_idx=cat_idx,
            **rf_args,
        ),
        "cforest": lambda: run_cforest(
            X,
            y,
            task=dataset.task,
            n_estimators=n_estimators,
            cat_idx=cat_idx,
        ),
    }

    importances: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    for method in methods:
        print(f"Running {method} ...", flush=True)
        started = time.perf_counter()
        result = functions[method]()
        values = np.asarray(result["importances"], dtype=float).reshape(-1)
        if values.size != len(dataset.feature_names):
            raise ValueError(
                f"{method} returned {values.size} importances for "
                f"{len(dataset.feature_names)} predictors"
            )
        importances[method] = values
        runtimes[method] = float(result.get("runtime", time.perf_counter() - started))
        print(f"  finished in {runtimes[method]:.2f} seconds", flush=True)

    result = pd.DataFrame(
        {
            "variable": dataset.feature_names,
            "type": dataset.feature_types,
        }
    )
    for method in methods:
        result[f"{method}_importance"] = importances[method]
        result[f"{method}_rank"] = _descending_min_ranks(importances[method])
    sort_method = "rf" if "rf" in methods else methods[0]
    result = result.sort_values(
        [f"{sort_method}_rank", "variable"],
        kind="stable",
    ).reset_index(drop=True)
    return result, runtimes


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="assemble and save the mixed dataset without fitting importance methods",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--jitter-strength", type=float, default=0.0001)
    parser.add_argument("--random-seed", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="importance methods to run (default: all)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.999,
        help="drop later continuous variables when absolute correlation exceeds this; use 1 to disable",
    )
    parser.add_argument("--top", type=int, default=30)
    return parser


def run_application(
    dataset: MixedDataset,
    dataset_output: Path,
    importance_output: Path,
    args: argparse.Namespace,
) -> None:
    selected_methods = tuple(dict.fromkeys(args.methods))
    dataset.print_summary()
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.assembled_frame().to_csv(dataset_output, index=True)
    print(f"Saved assembled data: {dataset_output}")

    if args.prepare_only:
        return

    ranking, runtimes = run_importance_methods(
        dataset=dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        jitter_strength=args.jitter_strength,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        methods=selected_methods,
    )
    ranking.to_csv(importance_output, index=False)
    print(f"Saved importance and ranks: {importance_output}")
    print(f"Method runtimes (seconds): {runtimes}")

    actual_methods = list(runtimes)
    top = min(args.top, len(ranking))
    columns = ["variable", "type"]
    for method in actual_methods:
        columns.extend([f"{method}_importance", f"{method}_rank"])
    sort_method = "rf" if "rf" in actual_methods else actual_methods[0]
    print(f"\nTop {top} variables, ordered by {sort_method} importance:")
    print(ranking.loc[: top - 1, columns].to_string(index=False))
