# Tree-based feature-importance methods
"""Importance estimators compared in the paper.

* ``jitterRF``    -- random-forest MDI, optionally with one-time or per-tree jitter
                     added to the categorical columns.
* ``jitterXGB``   -- XGBoost gain importance, optionally with one-time jitter.
* ``run_ufi``     -- unbiased feature importance (Zhou & Hooker) from a random forest.
* ``run_cforest`` -- conditional random forest via R ``partykit`` (optional; needs
                     ``rpy2`` and the R packages ``partykit``, ``libcoin``, ``mvtnorm``).

Every function returns ``{'importances': np.ndarray, 'runtime': float, ...}``.
Jitter half-width defaults to 1e-4; see the manuscript, 'Jitter strength'.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from methods import ufi


_cforest_r_backend = None


def _get_cforest_r_backend():
    """Load rpy2 and the required R packages once, on first use."""
    global _cforest_r_backend
    if _cforest_r_backend is not None:
        return _cforest_r_backend

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as error:
        raise ImportError(
            'CForest requires rpy2 and a working R installation.'
        ) from error

    try:
        ro.r('''
            suppressPackageStartupMessages({
                library(partykit)
                library(libcoin)
                library(mvtnorm)
            })
        ''')
    except Exception as error:
        raise RuntimeError(
            'CForest requires the R packages partykit, libcoin, and mvtnorm.'
        ) from error

    _cforest_r_backend = (ro, pandas2ri, localconverter)
    return _cforest_r_backend


def unavailable_methods():
    """Method keys that cannot run in this environment.

    Everything is available except ``'cforest'`` when ``rpy2`` (and an R install) is
    missing. Callers use this to skip a baseline gracefully.
    """
    try:
        import rpy2  # noqa: F401
    except ImportError:
        return ('cforest',)
    return ()


#----------------------------------------------------------------
# Random forest
#----------------------------------------------------------------
def jitterRF(X, y, task='regression', jitter_method='per_tree', jitter_strength=0.0001,
             cat_idx=None, **rf_args):
    start_time = time.time()
    X = np.asarray(X)
    n, p = X.shape

    # which columns get jittered: caller-supplied indices, or every column that is
    # not all-distinct (the right rule for the simulation designs, where binary
    # predictors repeat and continuous ones do not).
    if cat_idx is None:
        cat_idx = [j for j in range(p) if np.unique(X[:, j]).size < n]
    p_cat = len(cat_idx)

    model_class = RandomForestRegressor if task == 'regression' else RandomForestClassifier
    variable_importances = np.zeros(p)

    def add_jitter(x_in):
        if p_cat == 0:
            return x_in
        x_jitter = x_in.astype(float, copy=True)
        noise = np.random.uniform(-jitter_strength, jitter_strength, size=(x_in.shape[0], p_cat))
        x_jitter[:, cat_idx] += noise
        return x_jitter

    if jitter_method is None:
        model = model_class(**rf_args)
        model.fit(X, y)
        variable_importances = model.feature_importances_

    elif jitter_method == 'one_time':
        model = model_class(**rf_args)
        model.fit(add_jitter(X), y)
        variable_importances = model.feature_importances_

    elif jitter_method == 'per_tree':
        n_estimators = rf_args.get('n_estimators', 100)
        rf_args_tree = rf_args.copy()
        rf_args_tree['n_estimators'] = 1
        rf_args_tree['n_jobs'] = None

        for _ in range(n_estimators):
            x_jittered = add_jitter(X)
            model = model_class(**rf_args_tree)
            model.fit(x_jittered, y)
            variable_importances += model.feature_importances_

        variable_importances /= n_estimators

    runtime = time.time() - start_time
    return {'importances': variable_importances, 'runtime': runtime}


#----------------------------------------------------------------
# XGBoost
#----------------------------------------------------------------
def jitterXGB(X, y, task='regression', jitter_method=None,
              jitter_strength=0.0001, max_depth=None, cat_idx=None):
    start_time = time.time()
    X = np.asarray(X)
    n, p = X.shape

    if cat_idx is None:
        cat_idx = [j for j in range(p) if np.unique(X[:, j]).size < n]

    x_fit = X
    if jitter_method == 'one_time' and len(cat_idx) > 0:
        x_fit = X.astype(float, copy=True)
        x_fit[:, cat_idx] += np.random.uniform(
            -jitter_strength,
            jitter_strength,
            size=(n, len(cat_idx)),
        )

    model_class = XGBRegressor if task == 'regression' else XGBClassifier
    model_args = {}
    if max_depth is not None:
        model_args['max_depth'] = max_depth
    model = model_class(**model_args)
    model.fit(x_fit, y)

    runtime = time.time() - start_time
    return {'importances': model.feature_importances_, 'runtime': runtime}


#----------------------------------------------------------------
# UFI
#----------------------------------------------------------------
def run_ufi(X, y, task='regression', n_estimators=100, cat_idx=None, **rf_args):
    # cat_idx is accepted for a uniform call signature with the jitter methods; UFI
    # is estimated from the forest structure and does not use it.
    start_time = time.time()
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, **rf_args)
        model.fit(X, y)
        importances = ufi.regr(model, X, y)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, **rf_args)
        model.fit(X, y)
        importances = ufi.cls(model, X, y)
    runtime = time.time() - start_time

    return {'error': 0, 'importances': importances, 'runtime': runtime}


#----------------------------------------------------------------
# Conditional random forest (R partykit)
#----------------------------------------------------------------
def run_cforest(X, y, task='regression', n_estimators=100, cat_idx=None, **cforest_args):
    # cat_idx is accepted for a uniform call signature; cforest infers categorical
    # predictors from the R factor columns built below.
    ro, pandas2ri, localconverter = _get_cforest_r_backend()
    start_time = time.time()

    feature_names = [f'X{i + 1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Y'] = y

    # R factors ensure that cforest treats binary predictors as categorical.
    for column in feature_names:
        if set(df[column].unique()) <= {0, 1}:
            df[column] = df[column].map({0: '0', 1: '1'}).astype('category')

    # cforest determines the task from the R class of the response.
    if task == 'classification':
        df['Y'] = df['Y'].astype(str).astype('category')

    replace = cforest_args.get('replace', False)
    fraction = cforest_args.get('fraction', 0.632)

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['df'] = ro.conversion.py2rpy(df)
    ro.globalenv['ntree'] = n_estimators
    ro.globalenv['replace'] = replace
    ro.globalenv['fraction'] = fraction

    ro.r('''
        cf <- cforest(
            Y ~ .,
            data = df,
            ntree = ntree,
            perturb = list(replace = replace, fraction = fraction)
        )
        imp <- varimp(cf)
        all_vars <- colnames(df)[colnames(df) != "Y"]
        full_imp <- setNames(rep(0, length(all_vars)), all_vars)
        full_imp[names(imp)] <- imp
    ''')

    variable_importances = np.asarray(ro.r('full_imp'), dtype=float)
    runtime = time.time() - start_time
    return {'error': 0, 'importances': variable_importances, 'runtime': runtime}
