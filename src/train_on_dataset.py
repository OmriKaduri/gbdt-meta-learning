import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import hyperopt

from src.metrics import evaluate_metrics
from src.HyperoptObjective import HyperoptObjective
from hyperopt import space_eval


# from PyFastBDT import FastBDT


def find_best_params(X_train,
                     y_train,
                     X_test,
                     y_test,
                     model,
                     const_params,
                     parameter_space,
                     fit_params={},
                     max_evals=25,
                     ovr=False
                     ):
    objective = HyperoptObjective(X_train, y_train, X_test, y_test, model, const_params,
                                  fit_params, ovr)
    '''
    HyperOpt Trials object stores details of every iteration. 
    https://github.com/hyperopt/hyperopt/wiki/FMin#12-attaching-extra-information-via-the-trials-object
    '''
    trials = hyperopt.Trials()

    '''
    Hyperopt fmin function returns only parameters from the search space.
    Therefore, before returning best_params
    we will merge best_params with the const params, 
    so we have all parameters in one place for training the best model.
    '''
    best_params = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.tpe.suggest,
        rstate=np.random.RandomState(seed=42),
        max_evals=max_evals,
        trials=trials
    )

    best_params = space_eval(parameter_space, best_params)
    best_params.update(const_params)
    return best_params, trials


def _get_model_and_params(model_name):
    if model_name == 'catboost':
        model = __import__('catboost').CatBoostClassifier
        const_params = {
            'verbose': False
        }
        parameter_space = {
            'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 250, 25)),
            'max_depth': hyperopt.hp.choice('max_depth', np.arange(5, 9)),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.5),
        }
        ovr = False
    elif model_name == 'fastbdt':
        model = __import__('PyFastBDT', fromlist=['FastBDT']).FastBDT.Classifier
        const_params = {}
        parameter_space = {
            'nTrees': hyperopt.hp.choice('n_estimators', np.arange(50, 250, 25)),
            'depth': hyperopt.hp.choice('max_depth', np.arange(5, 9))
        }
        ovr = True
    else:
        raise NotImplementedError

    return model, const_params, parameter_space, ovr


def inner_cv_hyperopt(X, y, model_name, n_splits=3):
    model, const_params, parameter_space, ovr = _get_model_and_params(model_name)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    best_auc = 0
    for index, (tr_ind, test_ind) in enumerate(kf.split(X, y)):
        X_train = X.iloc[tr_ind].copy()
        y_train = y.iloc[tr_ind].copy()

        X_test = X.iloc[test_ind].copy()
        y_test = y.iloc[test_ind].copy()
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns
        fit_params = {}
        if model_name == 'catboost':
            fit_params = {
                'cat_features': cat_features
            }

        curr_best_params, trials = find_best_params(
            X_train,
            y_train,
            X_test,
            y_test,
            model,
            const_params,
            parameter_space,
            fit_params,
            max_evals=50,
            ovr=ovr
        )

        fnvals = [(t['result']) for t in trials.trials]
        params = max(fnvals, key=lambda x: -x['loss'])
        if -params['loss'] > best_auc:
            best_auc = -params['loss']
            fit_time = params['fit_time']
            best_params = curr_best_params
            best_model = params['model']
            trials = trials

    return best_params, trials, fit_time, best_model


def outer_cv(X, y, results_df, record, n_splits=10, model_name='catboost'):
    df = results_df.copy()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for index, (tr_ind, tset_ind) in enumerate(kf.split(X, y)):
        print("Starting {i} fold out of {n} outher folds".format(i=index, n=n_splits))

        X_train = X.iloc[tr_ind].copy()
        y_train = y.iloc[tr_ind].copy()

        X_test = X.iloc[tset_ind].copy()
        y_test = y.iloc[tset_ind].copy()
        if set(y_train) - set(y_test) != {}:
            drop_indices = y_train[y_train.isin(list(set(y_train) - set(y_test)))].index
            y_train = y_train.drop(drop_indices)
            X_train = X_train.drop(drop_indices)
            y_train_labels = list(set(y_train))
            y_train_labels.sort()
            fix_labels = {k: v for v, k in enumerate(y_train_labels)}
            y_train = y_train.replace(fix_labels)
            y_test = y_test.replace(fix_labels)

        best_params, trials, fit_time, model = inner_cv_hyperopt(X_train, y_train, model_name=model_name)

        best_metrics = evaluate_metrics(model, X_test, y_test)
        best_metrics['Training_time'] = fit_time
        print("BEST: ", best_metrics)
        info = {'CV_fold': index,
                'HP_vals': {k: best_params[k] for k in list(best_params)[:3]},
                **best_metrics
                }
        record.update(info)
        df = df.append(record, ignore_index=True)
    return df
