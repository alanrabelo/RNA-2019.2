import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from Dataset.datasets import RegressionDatasets, ClassificationDatasets

class DataManager:

    def __init__(self, filename, validations_percentage=0.1, test_percentage=0.2):

        self.validations_percentage = validations_percentage
        self.test_percentage = test_percentage
        self.filename = filename
        self.X = []
        self.Y = []
        self.load_data()


    def load_data(self, categorical=True):

        with open('Dataset/' + self.filename.value, 'r') as file:
            self.X = []
            self.Y = []
            categorical = {}

            for line in file.readlines():
                if line == '' or '?' in line:
                    continue
                input_formatted = []

                line = line.replace('\n', '')
                split_line = line.split(' ' if self.filename == ClassificationDatasets.COLUNA else ',')

                output = split_line[-1]
                input = split_line[:-1]


                for value in input:
                    try:
                        formatted = float(value)
                        input_formatted.append(formatted)
                    except:
                        continue

                self.X.append(input_formatted)

                if output in categorical:
                    self.Y.append(categorical[output])
                else:
                    categorical[output] = len(categorical.keys())
                    self.Y.append(categorical[output])

            if self.filename == ClassificationDatasets.CANCER:
                self.X = list(np.array(self.X)[:, 1:])
            scaler = MinMaxScaler(feature_range=(0, 1))
            x = scaler.fit_transform(self.X)
            self.X = x

            if categorical:
                self.Y = self.one_hot_encoding(self.Y)

    def one_hot_encoding(self, y_labels):

        result = []
        values_dict = {}

        for label in y_labels:

            if not label in values_dict:
                output = np.zeros(len(set(y_labels)))
                output[len(values_dict.keys())] = 1
                values_dict[label] = output
            result.append(values_dict[label])

        return np.array(result, dtype=int)

    def split_train_test_5fold(self, X, Y, validation=False, hasvalidation=False):

        c = list(zip(np.array(X), np.array(Y)))
        random.shuffle(c)
        X_shuffled, Y_shuffled = zip(*c)

        x_TRAINS = []
        y_TRAINS = []
        x_TESTS = []
        y_TESTS = []

        validation_split_point = round(len(X) * self.validations_percentage) if validation else 0
        validation_X = X_shuffled[:validation_split_point]
        validation_Y = Y_shuffled[:validation_split_point]

        X_shuffled = X_shuffled[validation_split_point:]
        Y_shuffled = Y_shuffled[validation_split_point:]

        for i in range(0, 5):

            size_of_the_fold = round(len(X_shuffled)/5.0)
            split_point1 = i * size_of_the_fold
            split_point2 = split_point1 + size_of_the_fold
            x_TRAINS.append(np.array(X_shuffled[:split_point1] + X_shuffled[split_point2:]))
            y_TRAINS.append(np.array(Y_shuffled[:split_point1] + Y_shuffled[split_point2:]))
            x_TESTS.append(np.array(X_shuffled[split_point1:split_point2]))
            y_TESTS.append(np.array(Y_shuffled[split_point1:split_point2]))

        if hasvalidation:
            return np.array(x_TRAINS)[0], np.array(y_TRAINS)[0], np.array(x_TESTS)[0], np.array(y_TESTS)[0]
        else:
            return np.array(x_TRAINS), np.array(y_TRAINS), np.array(x_TESTS), np.array(y_TESTS)











