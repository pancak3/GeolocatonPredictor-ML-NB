import logging
from util import train
from util import predict
from util.preprocessing import merge


def merge_files():
    logging.basicConfig(level=logging.INFO)
    f_path = "datasets/dev-best200.csv"
    merge.merge(f_path)
    f_path = "datasets/train-best200.csv"
    merge.merge(f_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # merge_files()
    f_path = train.random_forest("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
                                 "datasets/dev-best200.csv")
    # predict.random_forest(f_path,
    #                       "myData/merged_dev-best200.csv", "datasets/dev-best200.csv")
