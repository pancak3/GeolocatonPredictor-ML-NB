import logging
import os
import re
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from joblib import dump
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from ..maps.maps import *


def feature_filter(features_list):
    res = []

    for feature in features_list:
        if len(feature) < 3:
            res.append(feature)
    return res


def merge_similar_feature(data, features):
    column = data[features].sum(axis=1)
    df = pd.DataFrame(data=column, columns=[features[0]])
    data = data.drop(columns=features)
    data.insert(2, features[0], df)
    return data


def merge(f_path):
    logging.info("[*] Merging %s " % f_path)
    data = pd.read_csv(f_path)
    features = feature_filter(data.columns[2:-1])
    data = data.drop(columns=features)

    data_features = data.columns[2:-1]
    features = []
    for feature in data_features:
        if 'aha' in feature:
            features.append(feature)

    features = [features]
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
    f_path = os.path.join("myData", "merged_" + filename)
    new_df.to_csv(f_path)
    map_path = os.path.join("myData", "merged_" + filename + ".map")
    dump(id_map, map_path)
    logging.info("[*] Saved %s; %s " % (f_path, map_path))
    return f_path


def word_type(data_path):
    data = pd.read_csv(data_path, encoding="ISO-8859-1")
    p_t = re.compile(r'@\w+|RT|[^\w ]')
    _columns = ["prep", "pp", "topic", "adj_adv", "verb", "at_user"]
    features = []
    for idx, row in tqdm(data.iloc[:, 2:].iterrows(), unit=' tweets',
                         total=data.shape[0]):

        prep, pp, topic, adj_adv, verb, at_user = 0, 0, 0, 0, 0, 0

        topic = row['tweet'].count('#')
        at_user = row['tweet'].count('@USER')

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
        features.append(line)
    df_features = pd.DataFrame(data=features, columns=_columns)
    filename = os.path.basename(data_path)[:-4]
    df_features.to_csv("myData/my_features_{}.csv".format(filename))
