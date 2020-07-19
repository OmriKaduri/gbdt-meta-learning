import numpy as np
from itertools import combinations
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score
from timeit import default_timer as timer


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


def evaluate_metrics(model, X_test, y_test):
    infer_start = timer()
    test_preds_proba = model.predict_proba(X_test)
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
