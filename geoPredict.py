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
    FEATURES_GAZE_NAME = 'features_gazetteer.csv'
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
        # self.gazetteer.update({'all': set(())})
        #
        # for gaze in self.gazetteer.items():
        #     if gaze[0] != 'all':
        #         self.gazetteer["all"].update(gaze[1])
        # self.gazetteer['all'].remove('')

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

    features = {
        "train": pd.DataFrame(data={}),
        "dev": pd.DataFrame(data={}),
    }

    def __init__(self):
        self.data.file_load()

    def select_user(self):
        self.features['user'] = self.data.train_set['user']

    def load_features(self):
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_NAME)
        self.features = pd.read_csv(f_path)
        pass

    def calc_gazetteer(self):
        # data = self.data.train_set
        def run(_self, tmp_data, f_path_, name):
            for gaze in tqdm.tqdm(_self.data.gazetteer['all'], unit=" gazes"):
                _self.features[name][gaze] = pd.DataFrame(np.zeros((tmp_data.shape[0], 1))).iloc[:, 0]
                for idx, tweet in enumerate(tmp_data['tweet']):
                    gazes = gaze.split()
                    tmp = set(tweet.split())
                    for token in gazes:
                        if token in tmp:
                            if gaze in tweet:
                                _self.features[name].loc[idx, gaze] = tweet.count(gaze)
                                break

            _self.features[name]['geotag'] = tmp_data['geotag']
            _self.features[name].to_csv(f_path_, index=False)
            logging.info("[*] Saved %s." % f_path_)

        logging.info("[*] Gathering gazetteer features in train set")
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_NAME)
        run(self, self.data.train_set, f_path, 'train')

        logging.info("[*] Gathering gazetteer features in dev set")
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_NAME[:-4] + '_dev.csv')
        run(self, self.data.dev_set, f_path, 'dev')
        # use O(1) to find first then use O(n)

    def get_gazetteer_in_tweets(self):
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.GAZETTEER_IN_TWEETS_PATH)
        res = set(())
        if os.stat(f_path).st_size == 0:
            logging.info("[*] Capturing gazetteers ...")

            f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)
            f = open(f_path, encoding='ISO-8859-1').read()

            for gazes in self.data.gazetteer.items():
                gazetteers = set(())
                for gaze in tqdm.tqdm(gazes[1], unit=' gazetteers'):
                    p = re.compile(gaze)
                    if gaze != '' and len(re.findall(p, f)) > 0:
                        gazetteers.add(gaze)

                gazetteers = dict.fromkeys(gazetteers)
                for gaze in gazetteers:
                    gazetteers[gaze] = len(re.findall(gaze, f))
                gazetteers = sorted(gazetteers, key=gazetteers.get)
                res = res | set(gazetteers[-50:])
        else:
            f = open(f_path).readlines()
            for gaze in f:
                res.add(gaze[:-1])
        res = sorted(res)
        self.data.gazetteer['all'] = res

        write_file(res, f_path)


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
    # file.geography_gazetteer()
    feature = Features()
    feature.get_gazetteer_in_tweets()
    # feature.select_user()
    # feature.load_features()
    feature.calc_gazetteer()
    print()
    # print(str(file.dev_set['tweet'][24]))
