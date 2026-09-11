"""Cleaning rules for the six datasets retained for the experiments.

The supported datasets are:

* ``cylinder_banding`` (OpenML 6332)
* ``forest_fires`` (OpenML 44962)
* ``student_performance`` (OpenML 44967)
* ``hepatitis`` (PMLB)
* ``saheart`` (PMLB)
* ``titanic`` (PMLB)

Dataset-specific cleaning is performed by ``load_and_clean_dataset``.  For the
descriptive feature-importance analysis, ``preprocess_full_dataset`` imputes
missing values and encodes categories using all available observations.

Schizophrenia and Steel Plates Faults are intentionally not supported here.
"""

import argparse
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from methods.dataset_types import classify_column


OPENML_DATASET_IDS: Mapping[str, int] = {
    "cylinder_banding": 6332,
    "forest_fires": 44962,
    "student_performance": 44967,
}

PMLB_DATASETS = {"hepatitis", "saheart", "titanic"}
SUPPORTED_DATASETS = tuple(OPENML_DATASET_IDS) + tuple(sorted(PMLB_DATASETS))

DEFAULT_PMLB_CACHE = Path(__file__).resolve().parent / "pmlb" / "pmlb_cache"
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class PreparedDataset:
    """A cleaned dataset plus its observed-value predictor schema."""

    key: str
    X: pd.DataFrame
    y: pd.Series
    task: str
    continuous_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError(f"Unknown task: {self.task!r}")
        overlap = set(self.continuous_cols) & set(self.categorical_cols)
        if overlap:
            raise ValueError(
                f"Schema for {self.key} assigns columns to both types: "
                f"{sorted(overlap)}"
            )
        expected = list(self.continuous_cols) + list(self.categorical_cols)
        missing = [column for column in expected if column not in self.X.columns]
        extras = [column for column in self.X.columns if column not in expected]
        if missing or extras:
            raise ValueError(
                f"Schema mismatch for {self.key}: missing={missing}, extras={extras}"
            )
        if len(self.X) != len(self.y):
            raise ValueError("X and y have different numbers of rows")


CYLINDER_IDENTIFIER_COLS = (
    "timestamp",
    "cylinder_number",
)

def _as_series(y: Any) -> pd.Series:
    series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
    series = series.reset_index(drop=True)
    series.name = series.name or "target"
    return series


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _classify_predictors(
    frame: pd.DataFrame,
    key: str,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Classify predictors from observed values and remove constants."""

    kinds = {
        column: classify_column(frame[column])[0]
        for column in frame.columns
    }
    constants = tuple(
        column for column in frame.columns if kinds[column] == "constant"
    )
    continuous = tuple(
        column for column in frame.columns if kinds[column] == "continuous"
    )
    categorical = tuple(
        column for column in frame.columns if kinds[column] == "categorical"
    )

    if not continuous or not categorical:
        raise ValueError(
            f"{key} must contain both continuous and categorical/discrete "
            "predictors after cleaning"
        )

    selected = frame.drop(columns=list(constants)).copy()
    selected = selected[list(continuous + categorical)]
    return selected, continuous, categorical, constants


def _binary_target(
    y: Any,
    key: str,
    explicit_mapping: Mapping[Any, int] | None = None,
) -> pd.Series:
    series = _as_series(y)
    if series.isna().any():
        raise ValueError(f"{key} has missing target values")

    if explicit_mapping is not None:
        mapped = series.map(explicit_mapping)
        if mapped.isna().any():
            unexpected = sorted(series[mapped.isna()].astype(str).unique())
            raise ValueError(f"Unexpected target values in {key}: {unexpected}")
        return mapped.astype(int).rename(series.name)

    values = list(pd.unique(series))
    if set(values) == {0, 1}:
        return _numeric(series).astype(int).rename(series.name)
    raise ValueError(f"Unexpected binary target values in {key}: {values}")


def clean_cylinder_banding(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Remove two identifiers, then classify the observed predictors."""

    frame = X.copy().reset_index(drop=True)
    frame = frame.drop(
        columns=list(CYLINDER_IDENTIFIER_COLS),
        errors="ignore",
    )
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "cylinder_banding"
    )

    target = _as_series(y)
    normalized = target.astype("string").str.strip().str.casefold()
    if set(normalized.unique()) == {"band", "noband"}:
        target = normalized.map({"noband": 0, "band": 1}).astype(int)
    else:
        target = _binary_target(target, "cylinder_banding")

    return PreparedDataset(
        key="cylinder_banding",
        X=frame,
        y=target,
        task="classification",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            "timestamp and cylinder_number removed as date/identifier fields.",
            f"Observed constant columns removed: {', '.join(constants)}.",
            "job_number and customer retained as specified by the supplied schema.",
            "Predictor types inferred from observed values with the 20-level rule.",
        ),
    )


def clean_forest_fires(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Keep the official predictors and original burned-area target."""

    frame = X.copy().reset_index(drop=True)
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "forest_fires"
    )

    area = _numeric(_as_series(y))
    if area.isna().any() or (area < 0).any():
        raise ValueError("forest_fires area must be non-missing and non-negative")

    return PreparedDataset(
        key="forest_fires",
        X=frame,
        y=area,
        task="regression",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            "All 12 official predictors retained, including rain.",
            "X, Y, and rain are categorical/discrete under the 20-level rule.",
            f"Observed constant columns removed: {', '.join(constants) or 'none'}.",
            "Exact duplicate rows retained.",
            "The original non-negative area target is retained without log transform.",
        ),
    )


def clean_student_performance(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Remove the prior-grade leakage variables G1 and G2."""

    frame = X.copy().reset_index(drop=True)
    frame = frame.drop(columns=["G1", "G2"], errors="ignore")
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "student_performance"
    )

    target = _numeric(_as_series(y))
    if target.isna().any():
        raise ValueError("student_performance has missing G3 values")

    return PreparedDataset(
        key="student_performance",
        X=frame,
        y=target,
        task="regression",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            "G1 and G2 removed to prevent leakage into final grade G3.",
            "Numeric predictors with at most 20 levels are categorical/discrete.",
            f"Observed constant columns removed: {', '.join(constants) or 'none'}.",
            "G3=0 is retained as a valid target value.",
        ),
    )


def clean_hepatitis(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Retain the published PMLB predictor values and recode the target."""

    frame = X.copy().reset_index(drop=True)
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "hepatitis"
    )
    target = _binary_target(
        y,
        "hepatitis",
        explicit_mapping={1: 0, 2: 1},
    )

    return PreparedDataset(
        key="hepatitis",
        X=frame,
        y=target,
        task="classification",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            "Loaded predictors classified directly, retaining their PMLB encoding.",
            "No encoded values restored to missing values or decoded to UCI measurements.",
            "Predictor types inferred from observed values with the 20-level rule.",
            f"Observed constant columns removed: {', '.join(constants) or 'none'}.",
            "Target recoded from {1, 2} to {0, 1}.",
        ),
    )


def clean_saheart(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Classify SAHeart predictors from their observed values."""

    frame = X.copy().reset_index(drop=True)
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "saheart"
    )
    target = _binary_target(y, "saheart")

    return PreparedDataset(
        key="saheart",
        X=frame,
        y=target,
        task="classification",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            f"Observed constant columns removed: {', '.join(constants) or 'none'}.",
            "No scaling, outlier deletion, or row deletion is applied.",
        ),
    )


def clean_titanic(X: pd.DataFrame, y: Any) -> PreparedDataset:
    """Retain duplicate rows and classify predictors from observed values."""

    frame = X.copy().reset_index(drop=True)
    frame, continuous, categorical, constants = _classify_predictors(
        frame, "titanic"
    )
    target = _binary_target(y, "titanic")

    return PreparedDataset(
        key="titanic",
        X=frame,
        y=target,
        task="classification",
        continuous_cols=continuous,
        categorical_cols=categorical,
        notes=(
            "Duplicate retained rows are not removed because identities were stripped.",
            "fare=0 is retained as a valid value.",
            "sibsp and parch are categorical/discrete under the 20-level rule.",
            f"Observed constant columns removed: {', '.join(constants) or 'none'}.",
        ),
    )


CLEANERS = {
    "cylinder_banding": clean_cylinder_banding,
    "forest_fires": clean_forest_fires,
    "student_performance": clean_student_performance,
    "hepatitis": clean_hepatitis,
    "saheart": clean_saheart,
    "titanic": clean_titanic,
}

ALIASES = {
    "cylinder": "cylinder_banding",
    "cylinder-banding": "cylinder_banding",
    "forest-fires": "forest_fires",
    "forestfires": "forest_fires",
    "student": "student_performance",
    "student-performance": "student_performance",
    "sa_heart": "saheart",
}


def _normalise_key(key: str) -> str:
    normalized = key.strip().lower().replace(" ", "_")
    return ALIASES.get(normalized, normalized)


def _load_openml(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    try:
        import openml
    except ImportError as exc:
        raise ImportError(
            "Loading OpenML data requires the 'openml' package."
        ) from exc

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    return X, _as_series(y)


def _load_pmlb(name: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    cached_file = cache_dir / name / f"{name}.tsv.gz"
    if cached_file.exists():
        frame = pd.read_csv(cached_file, sep="\t", compression="gzip")
    else:
        try:
            from pmlb import fetch_data
        except ImportError as exc:
            raise ImportError(
                "Loading an uncached PMLB dataset requires the 'pmlb' package."
            ) from exc
        frame = fetch_data(name, local_cache_dir=str(cache_dir))

    if "target" not in frame.columns:
        raise ValueError(f"PMLB dataset {name!r} has no 'target' column")
    return frame.drop(columns=["target"]), _as_series(frame["target"])


def load_and_clean_dataset(
    key: str,
    pmlb_cache_dir: str | Path = DEFAULT_PMLB_CACHE,
) -> PreparedDataset:
    """Load one of the six supported datasets and apply its cleaning rules."""

    normalized = _normalise_key(key)
    if normalized not in CLEANERS:
        raise ValueError(
            f"Unsupported dataset {key!r}; choose one of {SUPPORTED_DATASETS}"
        )
    if normalized in OPENML_DATASET_IDS:
        X, y = _load_openml(OPENML_DATASET_IDS[normalized])
    else:
        X, y = _load_pmlb(normalized, Path(pmlb_cache_dir))
    return CLEANERS[normalized](X, y)


def _categorical_values(series: pd.Series) -> pd.Series:
    """Normalize labels while preserving numeric category order."""

    non_missing = series.dropna()
    numeric_non_missing = pd.to_numeric(non_missing, errors="coerce")
    if numeric_non_missing.notna().all():
        values = _numeric(series).astype("object")
    else:
        values = series.astype("string").str.strip().str.casefold().astype("object")
    return values.where(pd.notna(values), MISSING_CATEGORY)


def _category_sort_key(value: Any) -> tuple[int, float, str]:
    """Sort numeric levels numerically, text levels lexically, and missing last."""

    if isinstance(value, str) and value == MISSING_CATEGORY:
        return 2, 0.0, ""
    if isinstance(value, Real) and not pd.isna(value):
        return 0, float(value), ""
    return 1, 0.0, str(value)


def preprocess_full_dataset(
    dataset: PreparedDataset,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Impute and encode all observations for descriptive importance."""

    columns = list(dataset.continuous_cols + dataset.categorical_cols)
    frame = dataset.X[columns].copy()
    output = pd.DataFrame(index=frame.index)

    if dataset.key == "titanic":
        fare = _numeric(frame["fare"])
        passenger_class = frame["class"]
        valid = fare.notna() & passenger_class.notna()
        titanic_fare_by_class = (
            fare[valid].groupby(passenger_class[valid]).median().to_dict()
        )

    for column in dataset.continuous_cols:
        values = _numeric(frame[column])
        fill_value = float(values.median())
        if dataset.key == "titanic" and column == "fare":
            class_fallback = frame["class"].map(titanic_fare_by_class)
            values = values.fillna(class_fallback)
        output[column] = values.fillna(fill_value).astype(float)

    for column in dataset.categorical_cols:
        if dataset.key == "hepatitis":
            # PMLB already supplies numeric codes; preserve even its 1/2 labels.
            output[column] = frame[column]
            continue
        values = _categorical_values(frame[column])
        categories = sorted(pd.unique(values), key=_category_sort_key)
        category_map = {
            category: index for index, category in enumerate(categories)
        }
        output[column] = values.map(category_map).astype(int)

    X_processed = output.reset_index(drop=True)
    cat_idx = list(range(len(dataset.continuous_cols), len(columns)))
    return X_processed, dataset.y.reset_index(drop=True), cat_idx


def _print_summary(dataset: PreparedDataset, preprocess_full: bool) -> None:
    print(f"dataset: {dataset.key}")
    print(f"task: {dataset.task}")
    print(f"rows: {len(dataset.X)}")
    print(f"predictors: {dataset.X.shape[1]}")
    print(f"continuous: {len(dataset.continuous_cols)}")
    print(f"categorical/discrete: {len(dataset.categorical_cols)}")
    missing_cells = int(dataset.X.isna().sum().sum())
    print(f"predictor missing cells before imputation: {missing_cells}")
    if preprocess_full:
        X_processed, _, cat_idx = preprocess_full_dataset(dataset)
        print(f"processed missing cells: {int(X_processed.isna().sum().sum())}")
        print(f"categorical/discrete indices: {cat_idx}")
    for note in dataset.notes:
        print(f"note: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--pmlb-cache-dir",
        type=Path,
        default=DEFAULT_PMLB_CACHE,
    )
    parser.add_argument(
        "--preprocess-full",
        action="store_true",
        help="also report the fully imputed and encoded data summary",
    )
    args = parser.parse_args()
    dataset = load_and_clean_dataset(args.dataset, args.pmlb_cache_dir)
    _print_summary(dataset, args.preprocess_full)


if __name__ == "__main__":
    main()
