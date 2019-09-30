import pandas as pd
from .maps.maps import *
from sklearn import metrics


def my_score(actual_y, predict_y):
    accuracy = metrics.accuracy_score(actual_y, predict_y)
    precision = metrics.precision_score(actual_y, predict_y, average=None)
    recall = metrics.recall_score(actual_y, predict_y, average=None)
    f_score = metrics.f1_score(actual_y, predict_y, average=None)

    scores = pd.DataFrame(data=[precision, recall, f_score],
                          index=['precision', 'recall', 'f_score'], columns=MAP).T
    weighted = [metrics.precision_score(actual_y, predict_y, average='weighted'),
                metrics.recall_score(actual_y, predict_y, average='weighted'),
                metrics.f1_score(actual_y, predict_y, average='weighted')]
    weighted = pd.DataFrame(data=weighted, index=['precision', 'recall', 'f_score'], columns=['weighted']).T
    scores = scores.append(weighted, ignore_index=False)

    return accuracy, scores
