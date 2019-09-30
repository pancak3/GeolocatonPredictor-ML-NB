import logging
import time
from util import train
from util import predict
from util.preprocessing import merge


def merge_files():
    paths = [["datasets/dev-best200.csv",
              "myData/dev_tweets.txt",
              "myData/my_features_dev_tweets.csv"],
             ["datasets/train-best200.csv",
              "myData/train_tweets.txt",
              "myData/my_features_train_tweets.csv"],
             ["datasets/test-best200.csv",
              "myData/test_tweets.txt",
              "myData/my_features_test_tweets.csv"]
             ]
    for f_path in paths:
        merge.word_type(f_path[1])
        merge.merge(f_path[0], f_path[2])


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # merge_files()

    # f_path = train.complement_nb("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
    #                              "datasets/dev-best200.csv")
    f_path = train.random_forest("myData/merged_train-best200.csv", "myData/merged_dev-best200.csv",
                                 "datasets/dev-best200.csv")
    print(time.time()-start)
