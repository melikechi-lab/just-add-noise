# Prediction-error variants of the jittered ensembles
"""``jitterRF`` / ``jitterXGB`` that fit on a train split and return the test error.

These differ from ``methods.jitter`` (which return only importances): here each
function takes ``(X_train, X_test, y_train, y_test)`` and reports misclassification
rate (classification) or mean squared error (regression). Used only by
``run_prediction.py`` for Figure S5.
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier


def _test_error(task, y_test, preds):
    if task == 'regression':
        return mean_squared_error(y_test, preds)
    return 1 - accuracy_score(y_test, preds)


def jitterRF(X_train, X_test, y_train, y_test, task='regression',
             jitter_method='per_tree', jitter_strength=0.01, cat_idx=None,
             cat_unique_threshold=None, **rf_args):
    start_time = time.time()
    n, p = X_train.shape

    if cat_idx is None:
        if cat_unique_threshold is None:
            cat_unique_threshold = p // 2
        cat_idx = [j for j in range(p) if np.unique(X_train[:, j]).size < cat_unique_threshold]
    p_cat = len(cat_idx)

    model_class = RandomForestRegressor if task == 'regression' else RandomForestClassifier
    variable_importances = np.zeros(p)

    def add_jitter(x):
        if p_cat == 0:
            return x
        x_jitter = x.astype(float, copy=True)
        x_jitter[:, cat_idx] += np.random.uniform(-jitter_strength, jitter_strength, size=(x.shape[0], p_cat))
        return x_jitter

    if jitter_method is None:
        model = model_class(**rf_args)
        model.fit(X_train, y_train)
        variable_importances = model.feature_importances_
        preds = model.predict(X_test)

    elif jitter_method == 'one_time':
        model = model_class(**rf_args)
        model.fit(add_jitter(X_train), y_train)
        variable_importances = model.feature_importances_
        preds = model.predict(X_test)

    elif jitter_method == 'per_tree':
        n_estimators = rf_args.get('n_estimators', 100)
        rf_args_tree = {**rf_args, 'n_estimators': 1}
        preds_all = np.zeros((X_test.shape[0], n_estimators))
        for i in range(n_estimators):
            model = model_class(**rf_args_tree)
            model.fit(add_jitter(X_train), y_train)
            variable_importances += model.feature_importances_
            preds_all[:, i] = model.predict(X_test)
        variable_importances /= n_estimators
        if task == 'regression':
            preds = preds_all.mean(axis=1)
        else:
            preds = (preds_all.mean(axis=1) >= 0.5).astype(int)

    error = _test_error(task, y_test, preds)
    runtime = time.time() - start_time
    return {'error': error, 'importances': variable_importances, 'runtime': runtime}


def jitterXGB(X_train, X_test, y_train, y_test, task='regression', n_estimators=100,
              jitter_method='per_tree', jitter_strength=0.01, cat_idx=None, **xgb_args):
    start_time = time.time()
    n, p = X_train.shape

    if cat_idx is None:
        cat_idx = [j for j in range(p) if np.unique(X_train[:, j]).size < n]
    p_cat = len(cat_idx)

    model_class = XGBRegressor if task == 'regression' else XGBClassifier
    variable_importances = np.zeros(p)

    def add_jitter(x):
        if p_cat == 0:
            return x
        x_jitter = x.astype(float, copy=True)
        x_jitter[:, cat_idx] += np.random.uniform(-jitter_strength, jitter_strength, size=(x.shape[0], p_cat))
        return x_jitter

    default_args = dict(n_estimators=n_estimators, verbosity=0)
    default_args.update(xgb_args)

    if jitter_method is None:
        model = model_class(**default_args)
        model.fit(X_train, y_train)
    elif jitter_method == 'one_time':
        model = model_class(**default_args)
        model.fit(add_jitter(X_train), y_train)
    variable_importances = model.feature_importances_

    error = _test_error(task, y_test, model.predict(X_test))
    runtime = time.time() - start_time
    return {'error': error, 'importances': variable_importances, 'runtime': runtime}
