import logging
import os
import re
import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from joblib import dump, load
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from ..maps.maps import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFpr
from sklearn.feature_selection import VarianceThreshold
from ..MyMertrics import *
from pprint import pprint


def feature_filter(features_list):
    res = []

    for feature in features_list:
        if len(feature) < 2:
            res.append(feature)
    return res


def merge_similar_feature(data, features):
    column = data[features].sum(axis=1)
    df = pd.DataFrame(data=column, columns=[features[0]])
    data = data.drop(columns=features)
    data.insert(2, features[0], df)
    return data


def merge(f_path):
    """

    :param f_path: file path
    :return: results path
    """

    logging.info("[*] Merging %s " % f_path)
    data = pd.read_csv(f_path)
    features = feature_filter(data.columns[2:-1])
    data = data.drop(columns=features)

    data_features = data.columns[2:-1]
    features = [[], [], [], []]
    for feature in data_features:
        if 'aha' in feature:
            features[0].append(feature)
        if 'lmao' in feature:
            features[1].append(feature)
        if 'lmf' in feature or "fao" in feature:
            features[2].append(feature)
        if 'jus' in feature:
            features[3].append(feature)
    features.append(["huh", "hun"])
    features.append(["taco", "tacos"])
    features.append(["icheated", "icheatedbecause"])
    features.append(["lt", "ltlt", "ltreply"])
    features.append(["mad", "madd"])

    # features.append(["huh", "hun"])
    # features.append(["flex", "flexin"])
    # features.append(["dam", "damn", 'da'])
    # features.append(["kno", "know", 'knw'])
    # features.append(["dat", "dats"])
    # # features.append(["gon", "gone"])
    # # features.append(["iono", "ion"])
    # features.append(["factaboutme", "factsaboutme"])
    # features.append(["taco", "tacos"])
    # features.append(["icheated", "icheatedbecause"])
    # features.append(["lt", "ltlt", "ltreply"])
    # features.append(["mad", "madd"])
    # features.append(["bt", "btwn"])
    # # features.append(["loll", "lolss", "lolsz"])
    # # features.append(["cali", "california"])

    for f in features:
        data = merge_similar_feature(data, f)

    data['class'] = data['class'].map(MAP)
    _columns = data.columns
    users = set(data['user-id'])
    new_df = DataFrame(columns=data.columns[2:])
    id_map = []
    for user_id in tqdm(users, unit=" users"):
        tmp_line = data.loc[data['user-id'] == user_id]
        id_map.append(tmp_line.index)
        # line = tmp_line.iloc[:, 2:-1].sum(axis=0)/len(tmp_line.index)
        line = tmp_line.iloc[:, 2:-1].sum(axis=0)
        line['class'] = data.loc[data['user-id'] == user_id]['class'].iloc[0]
        line_df = DataFrame(data=line, columns=[user_id]).T
        new_df = new_df.append(line_df)
    filename = os.path.basename(f_path)
    # features selector
    if "train" in filename:
        # classes = [0, 1, 2]
        # features_percentage = pd.DataFrame(columns=classes, index=new_df.columns[:-1])
        # for cls in classes:
        #     data_class = new_df[new_df["class"] == cls].iloc[:, :-1].sum(axis=0)
        #     data_class_df = pd.DataFrame(data_class, columns=[cls])
        #     features_percentage[cls] = data_class_df
        #
        # # features_sum = features_percentage.sum(axis=1)
        # features_std = features_percentage.std(axis=1)
        # features_mean = features_percentage.mean(axis=1)
        # keep_features_list = []
        # for items in features_std.iteritems():
        #     if items[1] < features_mean[items[0]]:
        #         keep_features_list.append(items[0])
        #
        # new_df = new_df.drop(columns=keep_features_list)

        # selector = SelectKBest(f_classif, k=200)
        # selector = VarianceThreshold(threshold=0.01)
        # selector = SelectFpr(f_classif, alpha=1.0e-3)
        selector = SelectFpr(chi2, alpha=1.0e-3)
        selector.fit_transform(new_df.iloc[:, :-1], new_df["class"].to_list())
        features_map = selector.get_support(indices=True)
        dump(features_map, "myData/features.map")
    else:
        features_map = load("myData/features.map")

    features_map = np.append(features_map, [new_df.shape[1] - 1])
    new_df = new_df.iloc[:, features_map]
    f_path = os.path.join("myData", "merged_" + filename)
    new_df.to_csv(f_path)
    map_path = os.path.join("myData", "merged_" + filename + ".map")
    dump(id_map, map_path)
    logging.info("[*] Saved %s; %s \n" % (f_path, map_path))
    return f_path


def word_type(data_path):
    """

    :param data_path: data path
    :return: None
    """
    logging.info("[*] Counting word type %s" % data_path)
    data = pd.read_csv(data_path, encoding="ISO-8859-1")
    p_t = re.compile(r'@\w+|RT|[^\w ]')
    _columns = ["prep", "pp", "topic", "adj_adv", "verb", "at_user"]
    # _columns = ["prep", "pp", "topic", "adj_adv", "verb", "at_user"]
    features = []
    for idx, row in tqdm(data.iloc[:, 2:].iterrows(), unit=' tweets',
                         total=data.shape[0]):

        prep, pp, topic, adj_adv, verb, at_user = 0, 0, 0, 0, 0, 0

        # topic = int(row['tweet'].count('#') > 0)
        # at_user = int(row['tweet'].count('@USER') > 0)
        topic = int(row['tweet'].count('#') > 0)
        at_user = int(row['tweet'].count('@USER') > 0)

        text = re.sub(p_t, '', row['tweet']).replace('_', ' ').lower()
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        for idx_, token in enumerate(tags):
            if token[1] in {'IN', 'TO'}:
                prep += 1
            if 'NN' in token[1]:
                pp += 1
            if 'JJ' in token[1]:
                adj_adv += 1
            if 'VB' in token[1]:
                verb += 1
        line = [prep, pp, topic, adj_adv, verb, at_user]
        # line = [prep, topic, adj_adv, verb, at_user]
        features.append(line)
    df_features = pd.DataFrame(data=features, columns=_columns)
    filename = os.path.basename(data_path)[:-4]
    wt_path = "myData/my_features_{}.csv".format(filename)
    df_features.to_csv(wt_path)
    logging.info("[*] Saved word type %s \n" % wt_path)


def result_combination(is_train=True):
    """

    :param is_train: is train default True
    :return: None
    """
    if is_train:
        sub_dir = "train"
    else:
        sub_dir = "predict"

    results = []
    for (t, t, filenames) in os.walk("results/" + sub_dir):
        for filename in filenames:
            res = pd.read_csv("results/" + sub_dir + "/" + filename)["class"]
            results.append(pd.DataFrame(res))
        break
    if len(filenames) > 1:
        logging.info("[*] Combining {} results and capturing the majority...".format(len(filenames)))
        concat_res = pd.concat(results, axis=1)
        final_res = []
        for i, row in tqdm(concat_res.iterrows(), unit=" rows", total=concat_res.shape[0]):
            final_res.append(row.map(MAP).mode()[0])
    else:
        final_res = results[0]

    predict_y = final_res

    if is_train:
        predict_y = predict_y["class"].map(MAP).tolist()
        actual_y = pd.read_csv("datasets/dev-best200.csv")['class'].map(MAP).to_list()
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
    else:
        final_res.to_csv("results/final_results.csv")
