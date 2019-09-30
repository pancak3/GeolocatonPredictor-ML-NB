import logging
import os
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

REMAP = {0: "California", 1: "NewYork", 2: "Georgia"}
MAP = {"California": 0, "NewYork": 1, "Georgia": 2}


def merge(f_path):
    logging.info("[*] Merging %s " % f_path)
    data = pd.read_csv(f_path)
    data['class'] = data['class'].map(MAP)
    _columns = data.columns
    users = set(data['user-id'])
    new_df = DataFrame(columns=data.columns[2:])
    for user_id in tqdm(users, unit=" users"):
        tmp_line = data.loc[data['user-id'] == user_id]
        line = tmp_line.iloc[:, 2:-1].sum(axis=0)
        line['class'] = data.loc[data['user-id'] == user_id]['class'].iloc[0]
        line_df = DataFrame(data=line, columns=[user_id]).T
        new_df = new_df.append(line_df)
    filename = os.path.basename(f_path)
    f_path = os.path.join("myData", "merged_" + filename)
    new_df.to_csv(f_path)
    logging.info("[*] Saved %s " % f_path)
    return f_path


def split(df, res):
    pass
