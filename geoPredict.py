import os
import pandas as pd


class Config:
    DATA_PATH = "datasets"
    MY_DATA_PATH = "myData"
    DEV_NAME = "dev_tweets.txt"
    TEST_NAME = "test_tweets.txt"
    TRAIN_NAME = "train_tweets.txt"
    SET_HEAD = "tweet_id,user,tweet,geotag\n".encode(encoding='UTF-8')


class FileLoader:
    train_set = None
    dev_set = None
    test_set = None
    config = Config()
    pass

    def __init__(self):
        self.file_fixer()

        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)
        self.train_set = pd.read_csv(f_path)
        f_path = os.path.join(self.config.MY_DATA_PATH, self.config.TRAIN_NAME)

        pass

    def file_fixer(self):
        def fix(filename):
            f_path = os.path.join(self.config.DATA_PATH, filename)
            f = open(f_path, encoding="ISO-8859-1")
            content = []
            for line in f.readlines():
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
                content.append(real_line[1:].encode(encoding='UTF-8'))
            f.close()
            f_path = os.path.join(self.config.MY_DATA_PATH, filename)
            f = open(f_path, mode='wb+')
            f.write(self.config.SET_HEAD)
            f.writelines(content)
            f.close()

        fix(self.config.TRAIN_NAME)
        fix(self.config.TEST_NAME)
        fix(self.config.DEV_NAME)


if __name__ == '__main__':
    file = FileLoader()
    # file.file_fixer()
    pass
