import os
from .file_manager import *
from .MyMertrics import *
from joblib import load


def predict(model_path, test_path,):
    """
    :param model_path: models path
    :param test_path: test set path
    :return:
    """
    dev_features = pd.read_csv(test_path)

    best_estimator = load_model(model_path).model
    res = best_estimator.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_path)

    print("\n{}\n{}\n".format(accuracy, scores))
