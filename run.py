import logging
import time
import os
import shutil
from util import train
from util import predict
from util.preprocessing import merge


def run_train(train_path, evaluate_path):
    """

    :param train_path: train set path
    :param evaluate_path: evaluation set path
    :return:
    """
    shutil.rmtree("models/")
    os.mkdir("models/", 0o755)

    merge.merge(train_path)
    merge.merge(evaluate_path)

    train_basename = os.path.basename(train_path)
    evaluate_basename = os.path.basename(evaluate_path)

    # f_path = train.decision_tree("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    # f_path = train.random_forest("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    for i in range(1):
        train.complement_nb("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
                            evaluate_path)


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    run_train("datasets/train-best200.csv", "datasets/dev-best200.csv")
    # merge.result_combination(True)
    # predict.predict("models", "datasets/dev-best200.csv")
    print("[*] Time cost: {0:.4f} seconds".format(time.time() - start))
