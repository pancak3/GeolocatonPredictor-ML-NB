import sys
from .file_manager import *
import logging
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from .maps.maps import *
from .MyMertrics import *
from joblib import load
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeRegressor


def random_forest(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': range(80, 120, 2),
        'max_features': [None]
        # 'bootstrap': [False],

    }

    # grid = GridSearchCV(clf, param_dist, cv=train_features.shape[1], scoring='accuracy', n_jobs=-1, verbose=2)
    grid = GridSearchCV(clf, param_dist, cv=20, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path)

    f_path = save_model(grid, accuracy, scores)
    return f_path


def complement_nb(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = ComplementNB()
    param_dist = {
        # 'alpha': [1.0e-10],
        'alpha': np.linspace(1.0e-10, 1.0e-10 * 100, 100),
        'norm': [False]
    }

    # grid = GridSearchCV(clf, param_dist, cv=train_features.shape[1], scoring='accuracy', n_jobs=-1, verbose=2)
    grid = GridSearchCV(clf, param_dist, cv=10, scoring='accuracy', n_jobs=1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path)

    f_path = save_model(grid, accuracy, scores)
    return f_path


def decision_tree(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    # clf = DecisionTreeRegressor()
    # param_dist = {
    #     'random_state': [0]
    # }
    #
    # grid = GridSearchCV(clf, param_dist, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
    # grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())
    #
    # res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])
    res = clf.predict(dev_features.iloc[:, 1:-1]).astype(int)

    accuracy, scores = my_score(res, test_original_path)

    # f_path = save_model(grid, accuracy, scores)
    # return f_path
    return ''
