import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
from scipy.special import softmax

from src.ResFGB.resfgb.models import ResFGB, get_hyperparams
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
import glob
from sklearn.model_selection import cross_validate
import hyperopt
from collections import Counter

#

from itertools import combinations
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def fpr_tpr(y_true, y_pred):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)  # Precision

    return fpr, tpr


def fpr_tpr_for_threshold(y_true, y_pred_probs, threshold=.5):
    n_classes = np.unique(y_true)
    num_paires = (len(n_classes) * (len(n_classes) - 1)) // 2
    fpr_scores = np.zeros(num_paires)
    tpr_scores = np.zeros(num_paires)
    for ix, (a, b) in enumerate(combinations(n_classes, 2)):  # one vs one
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        y_pred_a = np.where(y_pred_probs[ab_mask, a] >= threshold, 1, 0)
        y_pred_b = np.where(y_pred_probs[ab_mask, b] >= threshold, 1, 0)

        fpr_a, tpr_a = fpr_tpr(a_true, y_pred_a)
        fpr_b, tpr_b = fpr_tpr(b_true, y_pred_b)
        fpr_scores[ix] = (fpr_a + fpr_b) / 2
        tpr_scores[ix] = (tpr_a + tpr_b) / 2

    return np.average(fpr_scores), np.average(tpr_scores)

#
def multiclass_precision_recall_curve(y_true, y_pred_probs):
    n_classes = np.unique(y_true)
    num_paires = (len(n_classes) * (len(n_classes) - 1)) // 2
    ap_scores = np.zeros(num_paires)

    for ix, (a, b) in enumerate(combinations(n_classes, 2)):  # one vs one
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        a_ap = average_precision_score(a_true, y_pred_probs[ab_mask, a])
        b_ap = average_precision_score(b_true, y_pred_probs[ab_mask, b])
        ap_scores[ix] = (a_ap + b_ap) / 2

    return np.average(ap_scores)
#

def evaluate_metrics(model, X_test, y_test):
    infer_start = timer()
    model.evaluate(X_test, y_test)
    test_preds = model.predict(X_test)
    test_preds = MinMaxScaler().fit_transform(test_preds)
    test_preds_proba = softmax(test_preds)
    test_preds = np.argmax(test_preds_proba, axis=1)
    infer_end = timer()
    infer_time = infer_end - infer_start  # For all dataset
    infer_time_per_1000 = (infer_time / X_test.shape[0]) * 1000

    fpr, tpr = fpr_tpr_for_threshold(y_test, test_preds_proba)
    is_multiclass = len(np.unique(y_test)) > 2
    if is_multiclass:
        auc = roc_auc_score(y_test, test_preds_proba, multi_class='ovr', average='macro')
        precision = precision_score(y_test, test_preds, average='macro')
        pr_curve = multiclass_precision_recall_curve(y_test, test_preds_proba)  # Macro avg
    else:
        test_preds_proba = np.max(test_preds_proba, axis=1)
        auc = roc_auc_score(y_test, test_preds_proba)
        precision = precision_score(y_test, test_preds)
        pr_curve = average_precision_score(y_test, test_preds_proba)

    return {'Accuracy': accuracy_score(y_test, test_preds),
            'TPR': tpr,  # macro average
            'FPR': fpr,  # macro average
            'Precision': precision,
            'AUC': auc,
            'PR-Curve': pr_curve,
            'Inference_time': infer_time_per_1000
            }
#

from hyperopt import STATUS_OK
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score, roc_curve
from timeit import default_timer as timer


class HyperoptObjective(object):
    def __init__(self, X_train, y_train, X_test, y_test, model, const_params):
        self.evaluated_count = 0
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.constant_params = const_params
        if self.y_train.dtype == 'object':
            le = LabelEncoder()
            self.y_train = le.fit_transform(self.y_train)
            self.y_test = le.fit_transform(self.y_test)

    '''
    The way that HyperOpt fmin function works, is that on each evaluation 
    it calls given objective function. 
    Since we decided to declare our objective as class instead of a function,
    we will implement the evaluation logic inside the __call__ method.
    '''

    def __call__(self, hyper_params):
        hparams = get_hyperparams(self.constant_params['n_data'],
                                  self.constant_params['input_dim'],
                                  self.constant_params['n_class'])
        del self.constant_params['n_data'], self.constant_params['input_dim'], self.constant_params['n_class']
        hparams.update(hyper_params)
        hparams.update(self.constant_params)
        hparams['model_hparams']['eval_iters'] = 1
        hparams['model_hparams']['minibatch_size'] = 32
        hparams['model_hparams']['tune_eta'] = False
        hparams['model_hparams']['max_epoch'] = 10
        input_dim = hparams['model_hparams']['shape'][0]

        hparams['resblock_hparams']['eval_iters'] = 1
        hparams['resblock_hparams']['minibatch_size'] = 32
        hparams['resblock_hparams']['max_epoch'] = 10
        hparams['resblock_hparams']['shape'] = (input_dim,100,input_dim)
        hparams['resblock_hparams']['tune_eta'] = False
        hparams['max_iters'] = 2

        model = self.model(**hparams)
        fit_start = timer()
        model.fit(self.X_train.to_numpy(), self.y_train)
        fit_end = timer()
        fit_time = fit_end - fit_start

        self.evaluated_count += 1

        metrics = evaluate_metrics(model, self.X_test, self.y_test)
        #         print("Inner AUC:",metrics['AUC'])
        #         print("Inner Accuracy:",metrics['Accuracy'])

        return {
            'loss': -metrics['AUC'],
            'status': STATUS_OK,
            'fit_time': fit_time,
            'model': model
        }
        # NOTE: The negative sign is due to that fact that we optimize for accuracy,
        # therefore we want to minimize the negative acc

#

from hyperopt import space_eval


def find_best_params(X_train,
                     y_train,
                     X_test,
                     y_test,
                     model,
                     const_params,
                     parameter_space,
                     max_evals=25,
                     ):
    objective = HyperoptObjective(X_train, y_train, X_test, y_test, model, const_params)
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
        max_evals=max_evals,
        trials=trials
    )
    best_params = space_eval(parameter_space, best_params)
    best_params.update(const_params)

    return best_params, trials

#

def inner_cv_hyperopt(X, y, n_classes, n_splits=3):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    parameter_space = {
        'model_type': hyperopt.hp.choice('model_type', ['logistic', 'smooth_hinge']),
        'fg_eta': hyperopt.hp.uniform('fg_eta', 0.05, 0.4),
    }
    best_auc = 0

    for index, (tr_ind, test_ind) in enumerate(kf.split(X, y)):
        #         print("Starting {i} fold out of {n} inner folds".format(i=index,n=n_splits))

        X_train = X.iloc[tr_ind].copy()
        y_train = y[tr_ind]

        X_test = X.iloc[test_ind].copy()
        y_test = y[test_ind]
        (n_data, input_dim) = X_train.shape

        const_params = {
            'n_data': n_data,
            'max_iters': 50,
            'input_dim': input_dim,
            'n_class': n_classes
        }

        curr_best_params, trials = find_best_params(
            X_train,
            y_train,
            X_test,
            y_test,
            ResFGB,
            const_params,
            parameter_space,
            max_evals=50,
        )
        # 2
        fnvals = [(t['result']) for t in trials.trials]
        params = max(fnvals, key=lambda x: x['loss'])
        if -params['loss'] > best_auc:
            best_auc = -params['loss']
            fit_time = params['fit_time']
            best_params = curr_best_params
            model = params['model']
            trials = trials

    return best_params, trials, fit_time, model

#

def outer_cv(X, y, results_df, record, n_splits=10):
    df = results_df.copy()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    is_multiclass = len(np.unique(y)) > 2
    n_classes = len(np.unique(y))
    for index, (tr_ind, tset_ind) in enumerate(kf.split(X, y)):
        print("Starting {i} fold out of {n} outher folds".format(i=index, n=n_splits))

        X_train = X.iloc[tr_ind].copy()
        y_train = y[tr_ind]

        X_test = X.iloc[tset_ind].copy()
        y_test = y[tset_ind]
        best_params, trials, fit_time, model = inner_cv_hyperopt(X_train, y_train, n_classes)
        best_metrics = evaluate_metrics(model, X_test, y_test, is_multiclass)
        best_metrics['Training_time'] = fit_time
        info = {'CV_fold': index,
                'HP_vals': {k: best_params[k] for k in list(best_params)[:3]},
                **best_metrics
                }
        record.update(info)
        df = df.append(record, ignore_index=True)
    return df

#

from pathlib import Path
from sklearn import preprocessing

results_df = pd.DataFrame(
    columns=['Dataset', 'Algorithm', 'CV_fold', 'HP_vals', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve',
             'Training_time', 'Inference_time'])
RESULTS_FILENAME = "catboost-results.csv"
results_file = Path(RESULTS_FILENAME)
if results_file.is_file():
    results_df = pd.read_csv(RESULTS_FILENAME)

for cls_dataset in glob.glob(r'../classification_datasets/*.csv')[6:]:
    dataset_name = Path(cls_dataset).stem
    if dataset_name in results_df.Dataset.unique():
        print("Skipping already fitted dataset: {f}".format(f=dataset_name))
        continue
    print("Fitting on {f} dataset".format(f=dataset_name))
    record = {'Dataset': dataset_name, 'Algorithm': 'Catboost'}
    df = pd.read_csv(cls_dataset)
    X =  pd.get_dummies(df[df.columns[:-1]])

    for c in X.columns:
        if X[c].dtype == 'int64':
            X[c] = X[c].astype('int32')

    y = df[df.columns[-1]]

    le = preprocessing.LabelEncoder()
    y_transformed = le.fit_transform(y)

    if y_transformed.dtype == 'int64':
        y_transformed = y_transformed.astype('int32')

    results_df = outer_cv(X, y_transformed, results_df, record)
    results_df.to_csv(RESULTS_FILENAME, index=False)

