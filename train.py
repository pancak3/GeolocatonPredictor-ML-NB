import datetime
import logging
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from pprint import pprint, pformat
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class Model:
    best_params = None
    best_score = None
    scoring = None
    cv = None
    model = None
    scores = None

    def __init__(self, best_params, best_score, scoring, cv, model, scores):
        self.best_params = best_params
        self.best_score = best_score
        self.scoring = scoring
        self.cv = cv
        self.model = model
        self.scores = scores


def save_model(grid, test_acc, scores):
    model = Model(grid.best_params_, test_acc, grid.scoring, grid.cv, grid.best_params_, scores)
    f_path = 'models/{0:.4f}_{1}'.format(test_acc, f"{datetime.datetime.now():%Y-%m-%d_%H:%M}")
    dump(model, f_path)
    logging.info("[*] Saved {} \n {}".format(f_path, pformat(vars(model))))


def load_model(f_path):
    model = load(f_path)
    logging.info("[*] Loaded {} \n {}".format(f_path, pformat(vars(model))))
    return model


def run():
    REMAP = {0: "California", 1: "NewYork", 2: "Georgia"}
    MAP = {"California": 0, "NewYork": 1, "Georgia": 2}
    f_path = "myData/merge_train.csv"
    train_features = pd.read_csv(f_path)
    f_path = "myData/merge_dev.csv"
    dev_features = pd.read_csv(f_path)

    train = train_features.iloc[:, 1:-1]
    train_y = train_features['class'].to_list()
    dev = dev_features.iloc[:, 1:-1]
    dev_y = dev_features['class'].to_list()

    del train_features
    del dev_features

    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': range(100, 102, 2)
    }

    grid = GridSearchCV(clf, param_dist, cv=2, scoring='accuracy', n_jobs=1, verbose=2)

    grid.fit(train, train_y)

    best_estimator = grid.best_estimator_
    res = best_estimator.predict(dev)
    result = pd.DataFrame({})
    result["prediction"] = pd.DataFrame(res).iloc[:, 0].map(REMAP)
    result["actual"] = dev_y
    actual_y = dev_y
    predict_y = res

    accuracy = []
    precision = []
    recall = []
    f_score = []

    accuracy.append(metrics.accuracy_score(actual_y, predict_y))
    precision.append(metrics.precision_score(actual_y, predict_y, average=None))
    recall.append(metrics.recall_score(actual_y, predict_y, average=None))
    f_score.append(metrics.f1_score(actual_y, predict_y, average=None))
    # print(" accuracy: %f\n precision: %f\n recall: %f\n f_score: %f" % (
    #     accuracy[-1], precision[-1], recall[-1], f_score[-1]))
    print("accuracy: %f" % accuracy[0])
    scores = pd.DataFrame(data=[precision[0], recall[0], f_score[0]],
                          index=['precision', 'recall', 'f_score'], columns=MAP).T

    weighted = [metrics.precision_score(actual_y, predict_y, average='weighted'),
                metrics.recall_score(actual_y, predict_y, average='weighted'),
                metrics.f1_score(actual_y, predict_y, average='weighted')]
    weighted = pd.DataFrame(data=weighted, index=['precision', 'recall', 'f_score'], columns=['weighted']).T
    scores = scores.append(weighted, ignore_index=False)
    pprint(scores)
    save_model(grid, accuracy[0], scores)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
