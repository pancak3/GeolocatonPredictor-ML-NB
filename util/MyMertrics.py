import os
import logging
import datetime
import pandas as pd
from .maps.maps import *
from sklearn import metrics
from joblib import load
from pprint import pprint
from tqdm import tqdm


def save_res(res, test_original_path, acc, is_train):
    """

    :param res: prediction y
    :param test_original_path: test original path
    :param acc: accuracy, for evaluation
    :param is_train: is train or not
    :return: None
    """
    filename = 'res_' + os.path.basename(test_original_path)
    if is_train:
        df = pd.DataFrame(res)['class'].map(REMAP)
    else:
        df = pd.DataFrame(res, columns=['class'])['class'].map(REMAP)

    df = pd.DataFrame(df, columns=["class"])
    if is_train:
        res_path = "results/train/{2}_{1:.4f}_{0}".format(filename, acc, f"{datetime.datetime.now():%Y-%m-%d_%H:%M}")

    else:
        filename = filename.replace("_merged", '')
        res_path = "results/predict/{1}_{0}".format(filename, f"{datetime.datetime.now():%Y-%m-%d_%H:%M}")

    while os.path.exists(res_path):
        res_path = res_path[:-4] + '_.csv'
    df.to_csv(res_path)
    logging.info("[*] Saved %s" % res_path)


def my_score(predict_y, test_original_path, is_train=True):
    """

    :param predict_y: predict labels list
    :param test_original_path: test original path
    :param is_train: called by train
    :return: (accuracy,scores)
    """
    # full_res = predict_y.copy()
    # predict_y = predict_y.argmax(axis=1)
    filename = os.path.basename(test_original_path)

    map_path = os.path.join("myData", 'merged_' + filename + ".map")

    id_map = load(map_path)
    test_original_df = pd.read_csv(test_original_path)
    predict_y_mapped = test_original_df["class"]
    logging.info("[*] Remapping ...")
    for idx, value in tqdm(enumerate(predict_y), unit=" users", total=len(predict_y)):
        predict_y_mapped.update(pd.Series([value for i in range(len(id_map[idx]))], index=id_map[idx], dtype=int))

    if is_train:
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
        save_res(predict_y, test_original_path, accuracy, is_train)
        return accuracy, scores
    else:
        predict_y = predict_y_mapped.astype(int)
        save_res(predict_y, test_original_path, None, is_train)

        return None


def get_scores(pred_path, actual_path):
    """

    :param pred_path: prediction path
    :param actual_path: actual path
    :return: None
    """
    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)
    if type(pred["class"][0]) is not int:
        pred["class"] = pred["class"].map(MAP)
    actual_y, predict_y = actual["class"].map(MAP), pred["class"].to_list()
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
    print("[*] Accuracy: %s" % accuracy)
    pprint(scores)
