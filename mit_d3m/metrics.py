# -*- coding: utf-8 -*-

import warnings

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def accuracy(ground_truth, predicted):
    return metrics.accuracy_score(ground_truth, predicted)


def f1(ground_truth, predicted):
    return metrics.f1_score(ground_truth, predicted)


def f1_micro(ground_truth, predicted):
    return metrics.f1_score(ground_truth, predicted, average='micro')


def f1_macro(ground_truth, predicted):
    return metrics.f1_score(ground_truth, predicted, average='macro')


def roc_auc(ground_truth, predicted):
    return metrics.roc_auc_score(ground_truth, predicted)


def roc_auc_micro(ground_truth, predicted):
    ground_truth, predicted = _binarize(ground_truth, predicted)
    return metrics.roc_auc_score(ground_truth, predicted, average='micro')


def roc_auc_macro(ground_truth, predicted):
    ground_truth, predicted = _binarize(ground_truth, predicted)
    return metrics.roc_auc_score(ground_truth, predicted, average='macro')


def l2(ground_truth, predicted):
    return (metrics.mean_squared_error(ground_truth, predicted))**0.5


def avg_l2(ground_truth_l, predicted_l):
    l2_sum = 0.0
    count = 0
    for pair in zip(ground_truth_l, predicted_l):
        l2_sum += l2(pair[0], pair[1])
        count += 1
    return l2_sum / count


def l1(ground_truth, predicted):
    return metrics.mean_absolute_error(ground_truth, predicted)


def r2(ground_truth, predicted):
    return metrics.r2_score(ground_truth, predicted)


def norm_mut_info(ground_truth, predicted):
    return metrics.normalized_mutual_info_score(ground_truth, predicted)


def jacc_sim(ground_truth, predicted):
    return metrics.jaccard_similarity_score(ground_truth, predicted)


def mean_se(ground_truth, predicted):
    return metrics.mean_squared_error(ground_truth, predicted)


def _binarize(ground, pred):
    label_binarizer = LabelBinarizer()
    return label_binarizer.fit_transform(ground), label_binarizer.transform(pred)


# MIT LL defined these strings here:
# https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/blob/shared/documentation/problemSchema.md#performance-metrics
METRICS_DICT = {
    'accuracy': accuracy,
    'f1': f1,
    'f1Micro': f1_micro,
    'f1Macro': f1_macro,
    'rocAuc': roc_auc,
    'rocAucMicro': roc_auc_micro,
    'rocAucMacro': roc_auc_macro,
    'meanSquaredError': mean_se,
    'rootMeanSquaredError': l2,
    'rootMeanSquaredErrorAvg': avg_l2,
    'meanAbsoluteError': l1,
    'rSquared': r2,
    'normalizedMutualInformation': norm_mut_info,
    'jaccardSimilarityScore': jacc_sim
}
