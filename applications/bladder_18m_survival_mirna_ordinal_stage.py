# Bladder ordinal-stage survival importance
"""Bladder mixed-data classification: 18-month survival from miRNA.

The outcome combines overall-survival time and vital status. Survival through
day 548 is coded as 1; death on or before day 548 is coded as 0; patients
alive/censored before day 548 are excluded. Stage I and missing-stage patients
are excluded, leaving pathologic stage as a three-tier ordered predictor with
levels Stage_II, Stage_III, and Stage_IV. Age remains continuous alongside the
complete bladder miRNA panel.

RF, jittered RF, XGBoost, jittered XGBoost, depth-1 XGBoost (with and without
one-time jitter), UFI, and CForest are all run over seeds 1, 2, 3, 4, and 5.
IPSS is not run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from output_guard import guard_outputs
from mixed_importance_utils import (
    METHODS,
    MixedDataset,
    _descending_min_ranks,
    _encode_dataset,
    make_fixed_horizon_mortality,
    normalize_text,
    ordered_overlap,
    prepare_continuous_omics,
    read_feature_by_sample,
    require_columns,
    run_importance_methods,
)


SCRIPT_DIR = Path(__file__).resolve().parent
# clinical.txt and mirna.txt (LinkedOmics TCGA-BLCA feature-by-sample tables) must be
# downloaded here first. Generated CSVs are written back into the same folder.
DATA_DIR = SCRIPT_DIR / 'data' / 'bladder'
DATASET_OUTPUT = DATA_DIR / 'bladder_18m_survival_mirna_ordinal_stage.csv'
IMPORTANCE_OUTPUT = (
    DATA_DIR / 'bladder_18m_survival_mirna_ordinal_stage_importance_by_seed.csv'
)
STAGE_RANK_OUTPUT = (
    DATA_DIR / 'bladder_18m_survival_mirna_ordinal_stage_pathologic_stage_ranks.csv'
)

# 18 average Gregorian months = 547.875 days, rounded to the nearest day.
HORIZON_DAYS = 548
DEFAULT_SEEDS = np.arange(1, 51)
STAGE_ORDER = ['stageii', 'stageiii', 'stageiv']
STAGE_MAP = {
    'stageii': 'Stage_II',
    'stageiii': 'Stage_III',
    'stageiv': 'Stage_IV',
}
STAGE_VARIABLE = 'pathologic_stage'

# Depth-1 XGBoost variants, run in addition to the shared METHODS.
STUMP_METHODS = ('xgb_stump', 'xgb_stump_one_time')
ALL_METHODS = (*METHODS, *STUMP_METHODS)


def prepare_dataset(correlation_threshold: float = 0.999) -> MixedDataset:
    """Merge bladder clinical and miRNA data by TCGA patient ID."""
    clinical = read_feature_by_sample(DATA_DIR / 'clinical.txt')
    mirna = read_feature_by_sample(DATA_DIR / 'mirna.txt')
    require_columns(
        clinical,
        ['overall_survival', 'status', 'years_to_birth', 'pathologic_stage'],
        'bladder clinical',
    )

    # Patient IDs are the row indexes after transposing both source tables.
    panel_ids = ordered_overlap(clinical.index, mirna.index)
    clinical = clinical.loc[panel_ids].copy()

    raw_stage = normalize_text(clinical['pathologic_stage'])
    observed_stages = set(raw_stage.dropna().unique().tolist())
    expected_stages = set(STAGE_ORDER) | {'stagei'}
    unexpected_stages = sorted(observed_stages - expected_stages)
    if unexpected_stages:
        raise ValueError(f'Unexpected values in pathologic_stage: {unexpected_stages}')

    stage = raw_stage.map(STAGE_MAP).astype('string').rename(STAGE_VARIABLE)
    age = pd.to_numeric(clinical['years_to_birth'], errors='coerce').rename(
        'clinical::age_years'
    )
    mortality = make_fixed_horizon_mortality(
        clinical['overall_survival'], clinical['status'], HORIZON_DAYS
    )
    survival = (1 - mortality).rename('SURVIVED_18_MONTHS')

    valid = mortality.notna() & age.notna() & raw_stage.isin(STAGE_ORDER)
    analysis_ids = panel_ids[valid.to_numpy()]

    omics, panel_report = prepare_continuous_omics(
        mirna,
        panel_ids=panel_ids,
        analysis_ids=analysis_ids,
        prefix='miRNA',
        correlation_threshold=correlation_threshold,
    )
    continuous = pd.concat(
        [omics, age.loc[analysis_ids].astype(float).to_frame()],
        axis=1,
    )
    categorical = stage.loc[analysis_ids].to_frame()

    dataset = MixedDataset(
        name='bladder_18m_survival_mirna_ordinal_stage',
        task='classification',
        outcome_name='SURVIVED_18_MONTHS',
        outcome=survival.loc[analysis_ids].astype(int),
        continuous=continuous,
        categorical=categorical,
        panel_report=panel_report,
    )
    dataset.validate()
    return dataset


def run_stump_methods(
    dataset: MixedDataset,
    jitter_strength: float,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit depth-1 XGBoost importances, with and without one-time jitter.

    These mirror the 'xgb' and 'xgb_one_time' methods in
    mixed_importance_utils.run_importance_methods but force max_depth=1. They
    live here because max_depth is not per-method configurable in the shared
    METHODS machinery.
    """
    from methods.jitter import jitterXGB

    configs = {
        'xgb_stump': {'jitter_method': None},
        'xgb_stump_one_time': {
            'jitter_method': 'one_time',
            'jitter_strength': jitter_strength,
        },
    }

    X, y, cat_idx = _encode_dataset(dataset)
    np.random.seed(random_seed)

    importances: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    for method, config in configs.items():
        print(f'Running {method} ...', flush=True)
        result = jitterXGB(
            X,
            y,
            task=dataset.task,
            cat_idx=cat_idx,
            max_depth=1,
            **config,
        )
        values = np.asarray(result['importances'], dtype=float).reshape(-1)
        if values.size != len(dataset.feature_names):
            raise ValueError(
                f'{method} returned {values.size} importances for '
                f'{len(dataset.feature_names)} predictors'
            )
        importances[method] = values
        runtimes[method] = float(result['runtime'])
        print(f'  finished in {runtimes[method]:.2f} seconds', flush=True)
    return importances, runtimes


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='assemble and save the mixed dataset without fitting importance methods',
    )
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--jitter-strength', type=float, default=0.0001)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=DEFAULT_SEEDS,
        help='random seeds to run (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.999,
        help=(
            'drop later continuous variables when absolute correlation exceeds '
            'this; use 1 to disable'
        ),
    )
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite existing output files instead of aborting',
    )
    return parser


def _method_columns(methods: tuple[str, ...]) -> list[str]:
    """Return the [<m>_importance, <m>_rank] column names for the given methods."""
    columns: list[str] = []
    for method in methods:
        columns.extend([f'{method}_importance', f'{method}_rank'])
    return columns


def run_seeded_importance(dataset: MixedDataset, args: argparse.Namespace) -> None:
    """Run every importance method for each seed and save rank summaries."""
    seeds = list(dict.fromkeys(args.seeds))
    full_rankings = []
    stage_rows = []

    for seed in seeds:
        print()
        print(f'Running seed {seed} ...', flush=True)
        ranking, runtimes = run_importance_methods(
            dataset=dataset,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            jitter_strength=args.jitter_strength,
            random_seed=seed,
            n_jobs=args.n_jobs,
            methods=METHODS,
        )

        stump_importances, stump_runtimes = run_stump_methods(
            dataset=dataset,
            jitter_strength=args.jitter_strength,
            random_seed=seed,
        )
        stump_frame = pd.DataFrame({'variable': dataset.feature_names})
        for method in STUMP_METHODS:
            stump_frame[f'{method}_importance'] = stump_importances[method]
            stump_frame[f'{method}_rank'] = _descending_min_ranks(
                stump_importances[method]
            )
        ranking = ranking.merge(stump_frame, on='variable', how='left')
        runtimes.update(stump_runtimes)

        ranking = ranking.loc[:, ['variable', 'type', *_method_columns(ALL_METHODS)]]
        ranking.insert(0, 'seed', seed)
        full_rankings.append(ranking)

        stage_row = ranking.loc[ranking['variable'].eq(STAGE_VARIABLE)]
        if len(stage_row) != 1:
            raise ValueError(f'Expected one {STAGE_VARIABLE} row in ranking output')
        stage_record = {'seed': seed, 'variable': STAGE_VARIABLE}
        for method in ALL_METHODS:
            stage_record[f'{method}_importance'] = float(
                stage_row[f'{method}_importance'].iloc[0]
            )
            stage_record[f'{method}_rank'] = int(stage_row[f'{method}_rank'].iloc[0])
            stage_record[f'{method}_runtime_seconds'] = runtimes[method]
        stage_rows.append(stage_record)

    full_ranking = pd.concat(full_rankings, ignore_index=True)
    stage_ranking = pd.DataFrame(stage_rows)

    IMPORTANCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    full_ranking.to_csv(IMPORTANCE_OUTPUT, index=False)
    stage_ranking.to_csv(STAGE_RANK_OUTPUT, index=False)
    print(f'Saved importance and ranks by seed: {IMPORTANCE_OUTPUT}')
    print(f'Saved pathologic-stage rank summary: {STAGE_RANK_OUTPUT}')

    top = min(args.top, len(full_ranking))
    columns = ['seed', 'variable', 'type', *_method_columns(ALL_METHODS)]
    print()
    print(f'Pathologic-stage ranks across {len(seeds)} seeds:')
    print(stage_ranking.to_string(index=False))
    print()
    print(f'Top {top} seed-variable rows, ordered by seed then rf rank:')
    print(
        full_ranking.sort_values(
            ['seed', 'rf_rank', 'variable'],
            kind='stable',
        )
        .loc[:, columns]
        .head(top)
        .to_string(index=False)
    )


def main() -> None:
    """Run the ordinal-stage bladder survival application."""
    print()
    args = build_parser().parse_args()

    outputs = [DATASET_OUTPUT]
    if not args.prepare_only:
        outputs += [IMPORTANCE_OUTPUT, STAGE_RANK_OUTPUT]
    guard_outputs(outputs, force=args.force)

    dataset = prepare_dataset(args.correlation_threshold)
    dataset.print_summary()
    DATASET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.assembled_frame().to_csv(DATASET_OUTPUT, index=True)
    print(f'Saved assembled data: {DATASET_OUTPUT}')

    if args.prepare_only:
        return

    run_seeded_importance(dataset, args)


if __name__ == '__main__':
    main()
