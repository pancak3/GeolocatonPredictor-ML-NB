from .MyMertrics import *
from .preprocessing import merge
from joblib import load
from tqdm import tqdm


def predict(model_path, test_path):
    """
    :param model_path: models path
    :param test_path: test set path
    :return: None
    """
    test_basename = os.path.basename(test_path)

    dev_features = pd.read_csv("myData/merged_" + test_basename)

    for (t, t, filenames) in os.walk(model_path):
        logging.info("[*] Predicting {0} with {1} models".format(test_path, len(filenames)))

        for filename in tqdm(filenames, unit=" models"):
            best_estimator = load(os.path.join(model_path, filename)).model
            res = best_estimator.predict(dev_features.iloc[:, 1:-1])
            my_score(res, test_path, is_train=False)
        break

    merge.result_combination(is_train=False)
