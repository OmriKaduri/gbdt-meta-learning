import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import hyperopt

from src.metrics import evaluate_metrics
from src.HyperoptObjective import HyperoptObjective
from hyperopt import space_eval


def find_best_params(X_train,
                     y_train,
                     X_test,
                     y_test,
                     model,
                     const_params,
                     parameter_space,
                     fit_params={},
                     max_evals=25,
                     ):
    objective = HyperoptObjective(X_train, y_train, X_test, y_test, model, const_params,
                                  fit_params)
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


def inner_cv_hyperopt(X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True)
    const_params = {
        'verbose': False
    }
    parameter_space = {
        'n_estimators': hyperopt.hp.choice('n_estimators', np.arange(50, 250, 25)),
        'max_depth': hyperopt.hp.choice('max_depth', np.arange(5, 9)),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.5),
    }
    best_auc = 0
    for index, (tr_ind, test_ind) in enumerate(kf.split(X, y)):
        #         print("Starting {i} fold out of {n} inner folds".format(i=index,n=n_splits))

        X_train = X.iloc[tr_ind].copy()
        y_train = y.iloc[tr_ind].copy()

        X_test = X.iloc[test_ind].copy()
        y_test = y.iloc[test_ind].copy()
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns
        fit_params = {
            'cat_features': cat_features
        }

        curr_best_params, trials = find_best_params(
            X_train,
            y_train,
            X_test,
            y_test,
            CatBoostClassifier,
            const_params,
            parameter_space,
            fit_params,
            max_evals=50,
        )

        fnvals = [(t['result']) for t in trials.trials]
        params = max(fnvals, key=lambda x: x['loss'])
        if -params['loss'] > best_auc:
            best_auc = -params['loss']
            fit_time = params['fit_time']
            best_params = curr_best_params
            model = params['model']
            trials = trials

    return best_params, trials, fit_time, model


def outer_cv(X, y, results_df, record, n_splits=10):
    df = results_df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True)
    for index, (tr_ind, tset_ind) in enumerate(kf.split(X, y)):
        print("Starting {i} fold out of {n} outher folds".format(i=index, n=n_splits))

        X_train = X.iloc[tr_ind].copy()
        y_train = y.iloc[tr_ind].copy()

        X_test = X.iloc[tset_ind].copy()
        y_test = y.iloc[tset_ind].copy()
        best_params, trials, fit_time, model = inner_cv_hyperopt(X_train, y_train)

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
