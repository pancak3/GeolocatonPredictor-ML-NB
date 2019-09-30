import os
import logging
import datetime
import pandas as pd
from .maps.maps import *
from sklearn import metrics
from joblib import load


def save_res(res, test_original_path, acc):
    filename = 'res_' + os.path.basename(test_original_path)
    df = pd.DataFrame(res)['class'].map(REMAP)
    res_path = "results/{0}_{1:.4f}_{2}".format(filename, acc, f"{datetime.datetime.now():%Y-%m-%d_%H:%M}")
    df.to_csv(res_path)
    print("[*] Saved %s" % res_path)


def my_score(predict_y, test_original_path):
    filename = os.path.basename(test_original_path)
    map_path = os.path.join("myData", 'merged_' + filename + ".map")
    id_map = load(map_path)
    test_original_df = pd.read_csv(test_original_path)
    predict_y_mapped = test_original_df["class"]
    for idx, value in enumerate(predict_y):
        predict_y_mapped.update(pd.Series([value for i in range(len(id_map[idx]))], index=id_map[idx], dtype=int))
    predict_y = predict_y_mapped.astype(int)
    actual_y = pd.read_csv(test_original_path)['class'].map(MAP).to_list()
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
    save_res(predict_y, test_original_path, accuracy)
    return accuracy, scores
