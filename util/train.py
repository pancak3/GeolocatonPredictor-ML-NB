from numpy import linspace, argmax
from .file_manager import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from .MyMertrics import *
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


def random_forest(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = RandomForestClassifier()
    param_dist = {
        'n_estimators': range(130, 132, 2)
    }
    # grid = GridSearchCV(clf, param_dist, cv=train_features.shape[1], scoring='accuracy', n_jobs=-1, verbose=2)
    grid = GridSearchCV(clf, param_dist, cv=42, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path, True)

    f_path = save_model(grid, accuracy, scores)
    return f_path


def container(train_path, test_path, test_original_path):
    """

    :param train_path: training set path
    :param test_path: evaluation set path
    :param test_original_path: evaluation set original path
    :return: None
    """
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    classifier_container = [
        {"model": BernoulliNB(alpha=1.0e-10),
         "name": "BernoulliNB",
         "params": {
             "n_estimators": [102],
             "max_samples": [0.3],
             "max_features": [0.5],
             # "n_estimators": range(102, 106, 2),
             # "max_samples": linspace(0.2, 0.4, 3),
             # "max_features": linspace(0.3, 0.5, 3),
             "warm_start": [True]
         }},
        {"model": ComplementNB(alpha=1.0e-10),
         "name": "ComplementNB",
         "params": {
             "n_estimators": [106],
             "max_samples": [0.2],
             "max_features": [0.5],
             # "n_estimators": range(102, 104, 2),
             # "max_samples": linspace(0.1, 0.3, 3),
             # "max_features": linspace(0.4, 0.6, 3),
             "warm_start": [True]
         }},
        {"model": MultinomialNB(alpha=1.0e-10),
         "name": "MultinomialNB",
         "params": {
             "n_estimators": [102],
             "max_samples": [0.3],
             "max_features": [0.5],
             # "n_estimators": range(100, 110, 2),
             # "max_samples": linspace(0.1, 0.5, 5),
             # "max_features": linspace(0.2, 0.6, 5),
             "warm_start": [True]
         }},
        {"model": RandomForestClassifier(
            n_estimators=106,
            criterion="entropy"
        ),
            "name": "RandomForestClassifier",
            "params": {
                "n_jobs": [-1]
            }
        },
        {"model": DecisionTreeClassifier(
            criterion="entropy"
        ),
            "name": "DecisionTreeClassifier",
            "params": {
            }}
    ]
    results = []
    for idx, clf in enumerate(classifier_container):
        logging.info("[*] ({1}/{2}) Training with {0} ...".format(clf["name"], idx + 1, len(classifier_container)))
        bag = BaggingClassifier(base_estimator=clf["model"])
        # grid = RandomizedSearchCV(bag, param_dist, cv=42, n_iter=300, scoring='accuracy', n_jobs=-1, verbose=2, refit=True)
        grid = GridSearchCV(bag, clf["params"], cv=42, scoring='accuracy', n_jobs=-1, verbose=0, refit=True)
        grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())
        res = grid.best_estimator_.predict_proba(dev_features.iloc[:, 1:-1])
        results.append(res)
        # clf["model"].fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())
        # res = clf["model"].predict(dev_features.iloc[:, 1:-1])
        # grid.best_estimator_ = clf["model"]
        # grid.best_params_ = None
        # grid.scoring = None
        # grid.cv = None
        # accuracy, scores = my_score(res, test_original_path, True)
        accuracy, scores = my_score(argmax(res, axis=1), test_original_path, True)
        save_model(grid, accuracy, scores)
    ensemble_res = results[0]
    for i in range(1, len(results)):
        ensemble_res += results[i]
    ensemble_res = argmax(ensemble_res, axis=1).tolist()
    accuracy, scores = my_score(ensemble_res, test_original_path, True)
    print("[*] Accuracy: %f" % accuracy)
    pprint(scores)
    # save_model(grid, accuracy, scores)

    print()


def decision_tree(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = DecisionTreeClassifier()
    # clf = ExtraTreeClassifier()
    param_dist = {
        # "max_depth": [None],
        # "min_samples_split": range(2, 3),
        # "random_state": [0]
        "criterion": ["entropy", "gini"],
        "min_samples_split": linspace(1.0e-5, 0.5, 1),
        # "class_weight": [{0: 1, 1: 1, 2: 1}],
        "min_samples_leaf": linspace(1.0e-5, 0.5, 10),
        "min_impurity_decrease": linspace(1.0e-5, 1, 10),
        "presort": [True, False]
        # "max_depth":[]
    }

    grid = GridSearchCV(clf, param_dist, cv=10, scoring='accuracy', n_jobs=1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    # from sklearn.tree.export import export_text
    # r = export_text(grid.best_estimator_)
    # pprint(r)
    # exit(0)
    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path, True)

    f_path = save_model(grid, accuracy, scores)
    return f_path
