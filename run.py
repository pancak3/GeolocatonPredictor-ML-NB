import time
import sys
import argparse
from pandas import DataFrame
from util import train
from util import predict
from util.file_manager import *
from util.preprocessing import merge
from util.MyMertrics import get_scores
from pprint import pprint


def run_train(train_path, evaluate_path, models_num):
    """

    :param train_path: train set path
    :param evaluate_path: evaluation set path
    :param models_num: models num for results voting
    :return:
    """
    remake_dir("models/")
    remake_dir("results/train")

    # merge.merge(train_path)
    # merge.merge(evaluate_path)

    train_basename = os.path.basename(train_path)
    evaluate_basename = os.path.basename(evaluate_path)

    # f_path = train.decision_tree("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    # f_path = train.random_forest("myData/merged_" + train_basename, "myData/merged_" + evaluate_basename,
    #                              evaluate_path)
    logging.info("[*] Training on {}, evaluating on {}".format(train_path, evaluate_path))
    for i in range(int(models_num)):
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
    parser = argparse.ArgumentParser(
        description='This script is used for predicting geotag of tweets based on Complement Naive Bayes.')
    parser.add_argument('-t', '--train', type=str, nargs=3,
                        metavar=('train_set_path', 'evaluate_set_path', 'models_num'),
                        help='python3 run.py -t ${train_set_path} ${evaluate_set_path} ${models_num} \n models_num >= 3 ')

    parser.add_argument('-p', '--predict', type=str, nargs=2, metavar=('models_path', 'test_set_path'),
                        help='python3 run.py -p ${models_path} ${test_set_path}')

    parser.add_argument('-s', '--score', type=str, nargs=2, metavar=('prediction_path', 'actual_path'),
                        help='python3 run.py -p ${prediction result path} ${actual path}')
    args = parser.parse_args()
    # start = time.time()
    time_cost = {}
    is_arg_empty = True
    if args.train is not None and len(args.train) == 3 and int(args.train[2]) > 2:
        time_cost.update({"Train": time.time()})
        run_train(args.train[0], args.train[1], args.train[2])
        time_cost["Train"] = time.time() - time_cost["Train"]
        is_arg_empty = False
    if args.predict is not None and len(args.predict) == 2:
        time_cost.update({"Predict": time.time()})
        run_predict(args.predict[0], args.predict[1])
        time_cost["Predict"] = time.time() - time_cost["Predict"]
        is_arg_empty = False

    if args.score is not None and len(args.score) == 2:
        time_cost.update({"Score": time.time()})
        get_scores(args.score[0], args.score[1])
        time_cost["Score"] = time.time() - time_cost["Score"]
        is_arg_empty = False

    if is_arg_empty:
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info("[*] Time costs in seconds:")
    pprint(DataFrame(time_cost, index=["Time_cost"]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parse()
    # start = time.time()
    # logging.basicConfig(level=logging.INFO)
    # run_train("datasets/train-best200.csv", "datasets/dev-best200.csv")
    # run_predict("models", "datasets/dev-best200.csv")
    # get_scores("results/final_results.csv", "datasets/dev-best200.csv")
    # logging.info("[*] Time cost: {0:.2f} seconds".format(time.time() - start))
