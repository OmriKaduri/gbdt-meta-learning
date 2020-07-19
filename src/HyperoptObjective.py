from sklearn.preprocessing import LabelEncoder
from hyperopt import STATUS_OK
from .metrics import evaluate_metrics

from timeit import default_timer as timer


class HyperoptObjective(object):
    def __init__(self, x_train, y_train, x_test, y_test, model, const_params, fit_params):
        self.evaluated_count = 0
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.model = model
        self.constant_params = const_params
        self.fit_params = fit_params
        # if self.y_train.dtype == 'object':  # Handle categorical target variable
        #     le = LabelEncoder()
        #     self.y_train = le.fit_transform(self.y_train)
        #     self.y_test = le.fit_transform(self.y_test)

    def __call__(self, hyper_params):
        model = self.model(**hyper_params, **self.constant_params)
        fit_start = timer()
        best = model.fit(X=self.X_train, y=self.y_train, **self.fit_params)
        fit_end = timer()
        fit_time = fit_end - fit_start

        self.evaluated_count += 1

        metrics = evaluate_metrics(best, self.X_test, self.y_test)

        return {
            'loss': -metrics['AUC'],
            'status': STATUS_OK,
            'fit_time': fit_time,
            'model': best
        }
        # NOTE: The negative sign is due to that fact that we optimize for accuracy,
        # therefore we want to minimize the negative acc
