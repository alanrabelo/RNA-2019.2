import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from Dataset.datasets import RegressionDatasets, ClassificationDatasets
import pandas as pd

class DataManager:

    def __init__(self, filename, validations_percentage=0.1, test_percentage=0.2):

        self.validations_percentage = validations_percentage
        self.test_percentage = test_percentage
        self.filename = filename
        self.X = []
        self.Y = []
        self.load_data()


    def load_data(self, categorical=True):

        values = self.read_data(self.filename)
        self.X = []
        self.Y = []
        categorical = {}

        for value in values:

            if self.filename == RegressionDatasets.FUEL:
                output = value[1]
                input = list(value[1:])
                input.append(value[0])
                input = np.array(input)
            else:
                output = value[-1]
                input = value[:-1]

            input_formatted = []

            for value in input:
                try:
                    formatted = float(value)
                    input_formatted.append(formatted)
                except:
                    continue

            self.X.append(input_formatted)

            if categorical:
                if output in categorical:
                    self.Y.append(categorical[output])
                else:
                    categorical[output] = len(categorical.keys())
                    self.Y.append(categorical[output])
            else:
                self.Y.append(output)

            if self.filename == ClassificationDatasets.CANCER:
                self.X = list(np.array(self.X)[:, 1:])

            self.X = self.scale(self.X)

            if categorical:
                self.Y = self.one_hot_encoding(self.Y)


    def read_data(self, filename):
        if 'csv' in self.filename.value:

            if filename == RegressionDatasets.FUEL:
                df = pd.read_csv('Dataset/' + self.filename.value, sep=',', decimal=',', header=0)
                df = df.dropna(axis=1, how="any")
                df['gas_type'] = pd.Categorical(df.gas_type).codes
            elif filename == RegressionDatasets.ABALONE:
                df = pd.read_csv('Dataset/' + self.filename.value, sep=',', header=None)
                df.dropna(axis='columns')
            return df.values

        else:
            with open('Dataset/' + self.filename.value, 'r') as file:

                values = []

                for line in file.readlines():
                    if line == '' or '?' in line:
                        continue
                    input_formatted = []

                    line = line.replace('\n', '')
                    split_line = line.split(' ' if self.filename == ClassificationDatasets.COLUNA else ',')
                    values.append(split_line)

                return values

    def scale(self, X):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = scaler.fit_transform(X)
        return list(x_scaled)

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











