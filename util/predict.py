from numpy import argmax
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
    results = []
    for (t, t, filenames) in os.walk(model_path):
        logging.info("[*] Predicting {0} with {1} models".format(test_path, len(filenames)))

        for filename in filenames:
            best_estimator = load(os.path.join(model_path, filename)).model
            res_prob = best_estimator.predict_proba(dev_features.iloc[:, 1:-1])
            results.append(res_prob)
            my_score(argmax(res_prob, axis=1), test_path, is_train=False)
        break
    ensemble_res = results[0]
    for i in range(1, len(results)):
        ensemble_res += results[i]
    ensemble_res = argmax(ensemble_res, axis=1).tolist()
    final_res = pd.DataFrame(ensemble_res, columns=["class"])
    final_res = pd.DataFrame(final_res["class"].map(REMAP), columns=["class"])

    filename = os.path.basename(test_path)
    map_path = os.path.join("myData", 'merged_' + filename + ".map")

    id_map = load(map_path)
    test_original_df = pd.read_csv(test_path)
    predict_y_mapped = test_original_df["class"]
    logging.info("[*] Remapping ...")
    for idx, value in tqdm(enumerate(final_res["class"]), unit=" users", total=len(final_res)):
        predict_y_mapped.update(pd.Series([value for i in range(len(id_map[idx]))], index=id_map[idx]))

    tweet_id = pd.DataFrame(data=test_original_df["tweet-id"], columns=["tweet-id"], dtype=int)
    tweet_class = pd.DataFrame(data=predict_y_mapped, columns=["class"])

    pd.concat([tweet_id, tweet_class], axis=1).to_csv("results/final_results.csv", index=False)
    print("\n")
    logging.info("[*] Saved results/final_results.csv")
