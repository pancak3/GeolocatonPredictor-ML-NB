import time
import sys
import argparse
from util import train
from util import predict
from util.file_manager import *
from util.preprocessing import merge
from util.MyMertrics import get_scores


def run_train(train_path, evaluate_path):
    """

    :param train_path: train set path
    :param evaluate_path: evaluation set path
    :return:
    """
    remake_dir("models/")
    remake_dir("results/train")

    merge.merge(train_path)
    merge.merge(evaluate_path)

    train_basename = os.path.basename(train_path)
    evaluate_basename = os.path.basename(evaluate_path)

    # f_path = train.decision_tree("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    # f_path = train.random_forest("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    logging.info("[*] Training on {}, evaluating on {}".format(train_path, evaluate_path))
    for i in range(42):
        train.complement_nb("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
                            evaluate_path)
    merge.result_combination(is_train=True)


def run_predict(models_path, test_path):
    """

    :param models_path: models path
    :param test_path: test set path
    :return: None
    """
    remake_dir("results/predict")

    merge.merge(test_path)

    predict.predict(models_path, test_path)
    merge.result_combination(is_train=False)


def arg_parse():
    parser = argparse.ArgumentParser(description='Predict geotag of tweets based on Complement Naive Bayes.')
    parser.add_argument('-t', '--train', type=str, nargs=2,
                        help='python3 run.py -t ${train_set_path} ${evaluate_set_path}')

    parser.add_argument('-p', '--predict', type=str, nargs=2,
                        help='python3 run.py -p ${models_path} ${test_set_path}')

    parser.add_argument('-s', '--score', type=str, nargs=2,
                        help='python3 run.py -p ${prediction_path} ${actual_path}')
    args = parser.parse_args()

    if 'train' in args:
        run_train(args.train[0], args.train[1])
    if 'predict' in args:
        run_predict(args.train[0], args.train[1])
    if 'score' in args:
        get_scores(args.train[0], args.train[1])
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_parse()
    # start = time.time()
    # logging.basicConfig(level=logging.INFO)
    # run_train("datasets/train-best200.csv", "datasets/dev-best200.csv")
    # run_predict("models", "datasets/dev-best200.csv")
    # get_scores("results/final_results.csv", "datasets/dev-best200.csv")
    # logging.info("[*] Time cost: {0:.2f} seconds".format(time.time() - start))
