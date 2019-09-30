import os
from .file_manager import *
from .MyMertrics import *
from joblib import load


def random_forest(model_path, test_path, test_original_path):
    dev_features = pd.read_csv(test_path)

    best_estimator = load_model(model_path).model
    res = best_estimator.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path)

    print("\n{}\n{}\n".format(accuracy, scores))
