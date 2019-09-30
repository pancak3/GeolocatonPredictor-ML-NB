from .file_manager import *
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from .maps.maps import *
from .MyMertrics import *


def random_forest(train_path, test_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': range(100, 102, 2)
    }

    grid = GridSearchCV(clf, param_dist, cv=2, scoring='accuracy', n_jobs=1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(dev_features['class'].to_list(), res)

    f_path = save_model(grid, accuracy, scores)
    return f_path
