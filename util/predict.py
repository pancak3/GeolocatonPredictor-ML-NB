from .file_manager import *
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from .maps.maps import *
from .MyMertrics import *


def random_forest(model_path, test_path):
    dev_features = pd.read_csv(test_path)

    best_estimator = load_model(model_path).model
    res = best_estimator.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(dev_features['class'].to_list(), res)

    print("\n{}\n{}\n".format(accuracy, scores))
