from .file_manager import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from .MyMertrics import *
from sklearn.naive_bayes import ComplementNB
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


def complement_nb(train_path, test_path, test_original_path):
    """

    :param train_path: training set path
    :param test_path: evaluation set path
    :param test_original_path: evaluation set original path
    :return: model path
    """
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)
    clf = ComplementNB(alpha=1.0e-10)
    param_dist = {
        "n_estimators": [106],
        "max_samples": [0.2],
        "max_features": [0.5],
        # "n_estimators": range(100, 110, 2),
        # "max_samples": np.linspace(0.1, 0.4, 4),
        # "max_features": np.linspace(0.3, 0.7, 5),
        "warm_start": [True]
    }
    bag = BaggingClassifier(base_estimator=clf)

    # grid = RandomizedSearchCV(bag, param_dist, cv=42, n_iter=300, scoring='accuracy', n_jobs=-1, verbose=2, refit=True)
    grid = GridSearchCV(bag, param_dist, cv=42, scoring='accuracy', n_jobs=-1, verbose=0, refit=True)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())
    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])
    accuracy, scores = my_score(res, test_original_path, True)
    f_path = save_model(grid, accuracy, scores)
    # pprint(train_features.shape)
    return f_path


def decision_tree(train_path, test_path, test_original_path):
    train_features = pd.read_csv(train_path)
    dev_features = pd.read_csv(test_path)

    clf = DecisionTreeClassifier()
    # clf = ExtraTreeClassifier()
    param_dist = {
        "max_depth": [None],
        "min_samples_split": range(2, 3),
        "random_state": [0]
    }

    grid = GridSearchCV(clf, param_dist, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(train_features.iloc[:, 1:-1], train_features['class'].to_list())

    res = grid.best_estimator_.predict(dev_features.iloc[:, 1:-1])

    accuracy, scores = my_score(res, test_original_path, True)

    f_path = save_model(grid, accuracy, scores)
    return f_path
