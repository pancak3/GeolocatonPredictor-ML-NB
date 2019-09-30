import logging
import datetime
from pprint import pprint, pformat
from joblib import dump, load
from .MyClasses import Model


def save_model(grid, test_acc, scores):
    model = Model(grid.best_params_, test_acc, grid.scoring, grid.cv, grid.best_estimator_, scores)
    f_path = 'models/{0:.4f}_{1}'.format(test_acc, f"{datetime.datetime.now():%Y-%m-%d_%H:%M}")
    dump(model, f_path)
    logging.info("\n{} \n[*] Saved {}".format(pformat(vars(model)), f_path))
    return f_path


def load_model(f_path):
    model = load(f_path)
    logging.info("\n{} \n[*] Loaded {}".format(pformat(vars(model)), f_path))
    return model



