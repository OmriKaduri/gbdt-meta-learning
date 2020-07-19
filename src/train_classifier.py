import glob
import traceback

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.train_on_dataset import outer_cv
import logging

logger = logging.getLogger('spam_application')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('train.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

results_df = pd.DataFrame(
    columns=['Dataset', 'Algorithm', 'CV_fold', 'HP_vals', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve',
             'Training_time', 'Inference_time'])

RESULTS_FILENAME = "../catboost-results.csv"
results_file = Path(RESULTS_FILENAME)
if results_file.is_file():
    results_df = pd.read_csv(RESULTS_FILENAME)

model_name = 'fastbdt'
# model_name = 'catboost'

for cls_dataset in glob.glob('../classification_datasets/*.csv'):
    dataset_name = Path(cls_dataset).stem
    if dataset_name in results_df.Dataset.unique():
        print("Skipping already fitted dataset: {f}".format(f=dataset_name))
        continue
    print("Fitting on {f} dataset".format(f=dataset_name))
    record = {'Dataset': dataset_name, 'Algorithm': model_name}
    df = pd.read_csv(cls_dataset)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    if y.dtype == 'object':  # Handle categorical target variable
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    try:
        results_df = outer_cv(X, y, results_df, record, model_name=model_name)
        results_df.to_csv(RESULTS_FILENAME, index=False)
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        logger.info("FAILED TO RUN ON {f}".format(f=dataset_name))
