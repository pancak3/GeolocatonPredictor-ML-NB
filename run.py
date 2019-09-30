import logging
from util import train
from util import predict
from util.preprocessing import merge


def merge_files():
    paths = ["datasets/dev-best200.csv",
             "datasets/train-best200.csv",
             "datasets/test-best200.csv"]
    for f_path in paths:
        merge.merge(f_path)


def count_type():
    paths = ["myData/dev_tweets.txt",
             "myData/train_tweets.txt",
             "myData/test_tweets.txt"]
    for f_path in paths:
        merge.word_type(f_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    merge_files()
    # f_path = train.random_forest("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
    #                              "datasets/dev-best200.csv")
    #

    # f_path = train.complement_nb("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
    #                              "datasets/dev-best200.csv")

    # f_path = train.decision_tree("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
    #                              "datasets/dev-best200.csv")
    # predict.random_forest(f_path,
    #                       "myData/merged_dev-best200.csv", "datasets/dev-best200.csv")
