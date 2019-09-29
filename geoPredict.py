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
from nltk import pos_tag
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


class Config:
    DATA_PATH = "datasets"
    MY_DATA_PATH = "myData"
    DEV_NAME = "dev_tweets.txt"
    TEST_NAME = "test_tweets.txt"
    TRAIN_NAME = "train_tweets.txt"
    SHP_PATH = "shapefiles"
    FEATURES_GAZE_TRAIN_NAME = 'features_gazetteer.csv'
    FEATURES_GAZE_DEV_NAME = 'features_gazetteer_dev.csv'
    GAZETTEER_IN_TWEETS_PATH = "gazetteer_in_tweets.txt"
    SHP = ("California", "NewYork", "Georgia")
    SHP_FILENAME = ("tl_2019_06_place", "tl_2019_36_place", "tl_2019_13_place")
    SET_HEAD = "tweet_id,user,tweet,geotag\n".encode(encoding='UTF-8')

    MAP = {"California": 0, "NewYork": 1, "Georgia": 2}
    REMAP = {0: "California", 1: "NewYork", 2: "Georgia"}


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
                # content.append(real_line[1:].encode(encoding="UTF-8"))
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
    best_most = []
    features = {
        "train": pd.DataFrame(data={}),
        "dev": pd.DataFrame(data={}),
    }
    nb_threshold = 1000

    def __init__(self):
        self.data.file_load()

    def select_user(self):
        self.features['user'] = self.data.train_set['user']

    def load_features(self):
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_TRAIN_NAME)
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

        logging.info("[*] Gathering gazetteer features in dev set")
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_DEV_NAME)
        run(self, self.data.dev_set, f_path, 'dev')

        logging.info("[*] Gathering gazetteer features in train set")
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_TRAIN_NAME)
        run(self, self.data.train_set, f_path, 'train')

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
                write_file(res, f_path)

        else:
            f = open(f_path).readlines()
            for gaze in f:
                res.add(gaze[:-1])

        res = sorted(res)
        self.data.gazetteer['all'] = res
        logging.info("[*] Loaded %d lines from %s.", len(res), f_path)

    def implement_paper(self):
        def run(_self, data):
            p_t = re.compile(r'@\w+|RT|[^\w ]')
            _columns = ["prep", "prep_pp", "pp", "defart_pp", "htah", "adj", "verb", "at_user"]
            features = []
            for idx, row in tqdm.tqdm(data.iloc[:, 2:].iterrows(), unit=' tweets',
                                      total=data.shape[0]):
                # character_set = set(row['tweet'])
                # if '@' in character_set:
                prep, prep_pp, pp, adj, verb, place_pp, defart_pp, time_expression, food = 0, 0, 0, 0, 0, 0, 0, 0, 0
                htah = row['tweet'].count('#')
                at_user = row['tweet'].count('@USER')

                text = re.sub(p_t, '', row['tweet']).replace('_', ' ').lower()
                tokens = word_tokenize(text)
                tags = pos_tag(tokens)
                for idx_, token in enumerate(tags):
                    if token[1] in {'IN', 'TO'}:
                        prep += 1
                        if idx_ < len(tags) - 2:
                            if 'NN' in tags[idx_ + 1][1]:
                                prep_pp += 1
                    if 'NN' in token[1]:
                        pp += 1
                        if idx_ > 0:
                            if tags[idx_ - 1][0] == 'the':
                                defart_pp += 1
                    if 'JJ' in token[1]:
                        adj += 1
                    if 'VB' in token[1]:
                        verb += 1
                line = [prep, prep_pp, pp, defart_pp, htah, adj, verb, at_user]
                features.append(line)
            df_features = pd.DataFrame(data=features, columns=_columns)
            df_features['geotag'] = data['geotag']
            return df_features

        def calc(_self):

            train_features = run(_self, _self.data.train_set)
            dev_features = run(_self, _self.data.dev_set)

            train_features.to_csv("myData/features_paper.csv")
            dev_features.to_csv("myData/features_paper_dev.csv")
            return train_features, dev_features

        f_path = "myData/features_paper.csv"
        if not os.path.exists(f_path):
            train_features, dev_features = calc(self)
        else:
            train_features, dev_features = pd.read_csv("myData/features_paper.csv"), pd.read_csv(
                "myData/features_paper_dev.csv")
            train = train_features
            train_y = train['geotag'].map(self.config.MAP)
            dev = dev_features

            accuracy = []
            precision = []
            recall = []
            f_score = []

            for t in range(1000, 1001):
                clf = ComplementNB(alpha=1000)
                # clf = RandomForestClassifier(n_estimators=1000)
                pprint(clf.fit(train.iloc[:, 1:-1], train_y))
                # cv_results = cross_validate(clf, train.iloc[:, 1:-1], train_y, cv=10)
                # pprint(cv_results.keys())
                #
                # pprint(pd.DataFrame(cv_results))

                # res = clf.predict(dev.iloc[:, 1:-1])
                res = cross_val_predict(clf, train.iloc[:, 1:-1], train_y, cv=9, n_jobs=-1)

                result = pd.DataFrame({})
                result["prediction"] = pd.DataFrame(res).iloc[:, 0].map(self.config.REMAP)
                result["actual"] = dev['geotag']
                actual_y = dev["geotag"].map(self.config.MAP)
                actual_y = train_y
                # pprint(result)
                predict_y = res

                accuracy.append(metrics.accuracy_score(actual_y, predict_y))
                precision.append(metrics.precision_score(actual_y, predict_y, average=None))
                recall.append(metrics.recall_score(actual_y, predict_y, average=None))
                f_score.append(metrics.f1_score(actual_y, predict_y, average=None))
                # print(" accuracy: %f\n precision: %f\n recall: %f\n f_score: %f" % (
                #     accuracy[-1], precision[-1], recall[-1], f_score[-1]))
                print("accuracy: %f" % accuracy[0])

                scores = pd.DataFrame(data=[precision[0], recall[0], f_score[0]],
                                      index=['precision', 'recall', 'f_score'], columns=self.config.MAP).T

                weighted = [metrics.precision_score(actual_y, predict_y, average='weighted'),
                            metrics.recall_score(actual_y, predict_y, average='weighted'),
                            metrics.f1_score(actual_y, predict_y, average='weighted')]
                weighted = pd.DataFrame(data=weighted, index=['precision', 'recall', 'f_score'], columns=['weighted']).T
                scores = scores.append(weighted, ignore_index=False)
                pprint(scores)

            # plt.plot(r, accuracy, '_', r, precision, '_', r, recall, '_', r, f_score, '_')
            # plt.show()
            # print(" accuracy: %f\n precision: %f\n recall: %f\n f_score: %f" % (accuracy, precision, recall, f_score))

    def paper_best_200(self):
        train_features, dev_features = pd.read_csv("datasets/train-best200.csv"), pd.read_csv(
            "datasets/dev-best200.csv")
        my_train, my_dev = pd.read_csv("myData/features_paper.csv"), pd.read_csv(
            "myData/features_paper_dev.csv")

        train_features = pd.concat(
            [train_features.iloc[:, 2:-1], my_train.iloc[:, 1:]], axis=1)
        dev_features = pd.concat([dev_features.iloc[:, 2:-1], my_dev.iloc[:, 1:]],
                                 axis=1)

        train = train_features
        train_y = train['geotag'].map(self.config.MAP)
        dev = dev_features

        accuracy = []
        precision = []
        recall = []
        f_score = []

        # clf = ComplementNB(alpha=self.nb_threshold)
        clf = RandomForestClassifier(n_estimators=1000)
        pprint(clf.fit(train.iloc[:, 2:-1], train_y))
        res = clf.predict(dev.iloc[:, 2:-1])

        result = pd.DataFrame({})
        result["prediction"] = pd.DataFrame(res).iloc[:, 0].map(self.config.REMAP)
        result["actual"] = dev['geotag']
        actual_y = dev["geotag"].map(self.config.MAP)
        # pprint(result)
        predict_y = res

        accuracy.append(metrics.accuracy_score(actual_y, predict_y))
        precision.append(metrics.precision_score(actual_y, predict_y, average=None))
        recall.append(metrics.recall_score(actual_y, predict_y, average=None))
        f_score.append(metrics.f1_score(actual_y, predict_y, average=None))
        # print(" accuracy: %f\n precision: %f\n recall: %f\n f_score: %f" % (
        #     accuracy[-1], precision[-1], recall[-1], f_score[-1]))
        print("accuracy: %f" % accuracy[0])
        scores = pd.DataFrame(data=[precision[0], recall[0], f_score[0]],
                              index=['precision', 'recall', 'f_score'], columns=self.config.MAP).T

        weighted = [metrics.precision_score(actual_y, predict_y, average='weighted'),
                    metrics.recall_score(actual_y, predict_y, average='weighted'),
                    metrics.f1_score(actual_y, predict_y, average='weighted')]
        weighted = pd.DataFrame(data=weighted, index=['precision', 'recall', 'f_score'], columns=['weighted']).T
        scores = scores.append(weighted, ignore_index=False)
        pprint(scores)

    def best_200(self):
        def merge(data):
            data['class'] = data['class'].map(self.config.MAP)
            _columns = data.columns
            users = set(data['user-id'])
            new_df = pd.DataFrame(columns=data.columns[2:])
            for user_id in tqdm.tqdm(users, unit=" users"):
                tmp_line = data.loc[data['user-id'] == user_id]
                line = tmp_line.iloc[:, 2:-1].sum(axis=0)
                line['class'] = data.loc[data['user-id'] == user_id]['class'].iloc[0]
                line_df = pd.DataFrame(data=line, columns=[user_id]).T
                new_df = new_df.append(line_df)

            return new_df

        train_features, dev_features = pd.read_csv("datasets/train-best200.csv"), pd.read_csv(
            "datasets/dev-best200.csv")
        train_features = merge(train_features)
        dev_features = merge(dev_features)

        train = train_features.iloc[:, :-1]
        train_y = train_features['class'].to_list()
        dev = dev_features.iloc[:, :-1]
        dev_y = dev_features['class'].to_list()

        # clf = ComplementNB(alpha=self.nb_threshold)
        clf = RandomForestClassifier(n_estimators=100)
        pprint(clf.fit(train, train_y))
        res = clf.predict(dev)
        # res = cross_val_predict(clf, train.iloc[:, 2:-1], train_y, cv=train.shape[1], n_jobs=-1)
        # res = cross_val_predict(clf, train.iloc[:, :-1], train_y, cv=2, n_jobs=-1)

        result = pd.DataFrame({})
        result["prediction"] = pd.DataFrame(res).iloc[:, 0].map(self.config.REMAP)
        result["actual"] = dev_y
        actual_y = dev_y
        # actual_y = train_y
        # pprint(result)
        predict_y = res

        accuracy = []
        precision = []
        recall = []
        f_score = []

        accuracy.append(metrics.accuracy_score(actual_y, predict_y))
        precision.append(metrics.precision_score(actual_y, predict_y, average=None))
        recall.append(metrics.recall_score(actual_y, predict_y, average=None))
        f_score.append(metrics.f1_score(actual_y, predict_y, average=None))
        # print(" accuracy: %f\n precision: %f\n recall: %f\n f_score: %f" % (
        #     accuracy[-1], precision[-1], recall[-1], f_score[-1]))
        print("accuracy: %f" % accuracy[0])
        scores = pd.DataFrame(data=[precision[0], recall[0], f_score[0]],
                              index=['precision', 'recall', 'f_score'], columns=self.config.MAP).T

        weighted = [metrics.precision_score(actual_y, predict_y, average='weighted'),
                    metrics.recall_score(actual_y, predict_y, average='weighted'),
                    metrics.f1_score(actual_y, predict_y, average='weighted')]
        weighted = pd.DataFrame(data=weighted, index=['precision', 'recall', 'f_score'], columns=['weighted']).T
        scores = scores.append(weighted, ignore_index=False)
        pprint(scores)

    def best_intersection_most(self):
        best = open('myData/attributes_name_best200.txt').readlines()
        most = open('myData/attributes_name_most200.txt').readlines()
        best = set(best)
        most = set(most)
        self.best_most = best.intersection(most)
        print()


class Predict:
    config = Config()

    def complement_naive_bayes(self):
        features = {}
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_TRAIN_NAME)
        features["train"] = pd.read_csv(f_path)
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.FEATURES_GAZE_DEV_NAME)
        features["dev"] = pd.read_csv(f_path)

        train = features["train"]
        train_y = train['geotag'].map(self.config.MAP)

        accuracy = []
        precision = []
        recall = []

        clf = MultinomialNB(alpha=0.01)
        pprint(clf.fit(train.iloc[:, :-1], train_y))
        dev = features["dev"]
        res = clf.predict(dev.iloc[:, :-1])
        result = pd.DataFrame({})
        result["prediction"] = pd.DataFrame(res).iloc[:, 0].map(self.config.REMAP)
        result["actual"] = dev['geotag']
        actual_y = dev["geotag"].map(self.config.MAP)
        predict_y = res

        accuracy.append(metrics.accuracy_score(actual_y, predict_y))
        precision.append(metrics.precision_score(actual_y, predict_y, average='macro'))
        recall.append(metrics.recall_score(actual_y, predict_y, average='macro'))


def write_file(contents, f_path):
    logger = logging.getLogger("write_file")
    f = open(f_path, encoding='utf-8', errors='ignore', mode='w+')
    for line in contents:
        f.writelines(line + "\n")
    f.close()
    logger.info("Wrote {} lines in {}.".format(len(contents), f_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # file = FileLoader()
    # file.file_fixer()
    # file.file_load()
    # file.freq_dist()
    # file.geography_gazetteer()
    feature = Features()
    # feature.get_gazetteer_in_tweets()
    # feature.select_user()
    # feature.load_features()
    # feature.calc_gazetteer()
    # feature.implement_paper()
    feature.best_200()
    # feature.best_intersection_most()
    # feature.paper_best_200()

    # predict = Predict()
    # predict.complement_naive_bayes()
    print()
    # print(str(file.dev_set['tweet'][24]))
