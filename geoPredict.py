import os
import errno
import logging
import re
import tqdm
import csv
import pandas as pd
import numpy as np
import shapefile as shp

from pprint import pprint
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


class Config:
    DATA_PATH = "datasets"
    MY_DATA_PATH = "myData"
    DEV_NAME = "dev_tweets.txt"
    TEST_NAME = "test_tweets.txt"
    TRAIN_NAME = "train_tweets.txt"
    SHP_PATH = "shapefiles"
    FEATURES_NAME = 'features.csv'
    GAZETTEER_IN_TWEETS_PATH = "gazetteer_in_tweets.txt"
    SHP = ("California", "NewYork", "Georgia")
    SHP_FILENAME = ("tl_2019_06_place", "tl_2019_36_place", "tl_2019_13_place")
    SET_HEAD = "tweet_id,user,tweet,geotag\n".encode(encoding='UTF-8')


class FileLoader:
    train_set = None
    dev_set = None
    test_set = None
    token_frequency = None
    gazetteer = {}
    config = Config()

    def file_load(self):
        # files = [self.config.DEV_NAME, self.config.TRAIN_NAME, self.config.TEST_NAME]
        # for filename in files:
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)
        self.train_set = pd.read_csv(f_path, encoding="ISO-8859-1")

        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TEST_NAME)
        self.test_set = pd.read_csv(f_path, encoding="ISO-8859-1")

        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.DEV_NAME)
        self.dev_set = pd.read_csv(f_path, encoding="ISO-8859-1")

        for idx, geotag in enumerate(self.config.SHP):
            f_path = os.path.join(self.config.MY_DATA_PATH, 'gazetteer_' + geotag + '.txt')
            self.gazetteer[geotag] = set(open(f_path).read().split('\n'))
        self.gazetteer.update({'all': set(())})

        for gaze in self.gazetteer.items():
            if gaze[0] != 'all':
                self.gazetteer["all"].update(gaze[1])
        self.gazetteer['all'].remove('')

    def file_fixer(self):
        def fix(filename):
            f_path = os.path.join(self.config.DATA_PATH, filename)
            f = open(f_path, encoding="ISO-8859-1")
            content = []
            for line in f.readlines():
                # line = line.decode('iso-8859-1').encode('utf8')
                columns = line.split(",")
                real_columns = ['', '', '', '']
                for idx, column in enumerate(columns):
                    if idx < 2:
                        real_columns[idx] = column
                    elif idx == len(columns) - 1:
                        real_columns[3] = column

                    else:
                        real_columns[2] += ',' + column
                real_columns[2] = real_columns[2][1:].replace('"', '""')[1:-1]
                real_line = ''
                for column in real_columns:
                    real_line += ',' + column
                content.append(real_line[1:].encode(encoding="ISO-8859-1"))
                # content.append(real_line[1:])
            f_path = os.path.join(self.config.MY_DATA_PATH, filename)

            '''
            @https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output/12517490
            '''
            try:
                os.makedirs(os.path.dirname(f_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

            f = open(f_path, mode='wb+')
            f.write(self.config.SET_HEAD)
            f.writelines(content)
            f.close()
            logger = logging.getLogger("file_fixer")
            logger.debug("[*] {} saved".format(f_path))

        fix(self.config.TRAIN_NAME)
        fix(self.config.TEST_NAME)
        fix(self.config.DEV_NAME)

    def freq_dist(self):
        def tweets_get_tokens(text):
            text = text.lower()

            # remove #topic @user
            p = re.compile("(#\w+|@\w+)")
            text = re.sub(p, '', text)

            # find all tokens that character repeated more than twice
            p = re.compile(r"[a-z]{,2}")
            text = re.sub(p, '', text)

            return text

        self.token_frequency = {
            'top200': {
            }
        }
        ny = self.train_set[self.train_set['geotag'] == "NewYork"]['tweet'].to_string()
        # tweets_get_tokens(ny)
        # pprint(ny)
        tmp = FreqDist(word.lower() for word in word_tokenize(ny))
        self.token_frequency['top200'].update({'NewYork': tmp})

    def geography_gazetteer(self):
        def shp_to_dict(_self, f_path_, geotag_):
            f_California = shp.Reader(f_path_)
            gazetteer = set(())
            for r in f_California.shapeRecords():
                gazetteer.add(r.record['NAME'])
            _self.gazetteer.update({geotag_: gazetteer})

        for idx, geotag in enumerate(self.config.SHP):
            f_path = os.path.join(self.config.MY_DATA_PATH, self.config.SHP_PATH, geotag, self.config.SHP_FILENAME[idx])
            shp_to_dict(self, f_path, geotag)
            f_path = os.path.join(self.config.MY_DATA_PATH, 'gazetteer_' + geotag + '.txt')

            write_file(self.gazetteer[geotag], f_path)


class Features:
    config = Config()
    data = FileLoader()

    features = pd.DataFrame(data={})

    def __init__(self):
        self.data.file_load()

    def select_user(self):
        self.features['user'] = self.data.train_set['user']

    def load_features(self):
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_NAME)
        self.features = pd.read_csv(f_path)
        pass

    def calc_gazetteer(self):

        for gaze in tqdm.tqdm(self.data.gazetteer['all'], unit=" gazes"):
            self.features[gaze] = pd.DataFrame(np.zeros((self.data.train_set.shape[0], 1))).iloc[:, 0]

        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)
        f = open(f_path)
        # use O(1) to find first then use O(n)
        for idx, tweet in enumerate(self.data.train_set['tweet']):
            self.features.loc[idx, gaze] = int(gaze in tweet)

    def get_gazetteer_in_tweets(self):
        logging.info("[*] Capturing gazetteers ...")
        gazetteers = set(())
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)
        f = open(f_path, encoding='ISO-8859-1').read()
        for gaze in tqdm.tqdm(self.data.gazetteer['all'], unit=' gazetteers'):
            p = re.compile(gaze)
            if gaze != '' and len(re.findall(p, f)) > 0:
                gazetteers.add(gaze)
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.GAZETTEER_IN_TWEETS_PATH)
        write_file(gazetteers, f_path)
        print()


def write_file(contents, f_path):
    logger = logging.getLogger("write_file")
    f = open(f_path, encoding='utf-8', errors='ignore', mode='w+')
    for line in contents:
        f.writelines(line + "\n")
    f.close()
    logger.info("Wrote {} lines in {}.".format(len(contents), f_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    file = FileLoader()
    # file.file_fixer()
    # file.file_load()
    # file.freq_dist()
    file.geography_gazetteer()
    feature = Features()
    # feature.get_gazetteer_in_tweets()
    # feature.select_user()
    # feature.load_features()
    feature.calc_gazetteer()
    print()
    # print(str(file.dev_set['tweet'][24]))
