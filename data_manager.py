import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataManager:



    def __init__(self, filename, validations_percentage=0.1, test_percentage=0.2):

        self.validations_percentage = validations_percentage
        self.test_percentage = test_percentage
        self.filename = filename
        self.X = []
        self.Y = []
        self.load_data()


    def load_data(self):

        with open('Dataset/' + self.filename, 'r') as file:
            self.X = []
            self.Y = []
            categorical = {}

            for line in file.readlines():

                input_formatted = []

                line = line.replace('\n', '')
                output = line.split(',')[-1]
                input = line.split(',')[:-1]

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


            scaler = MinMaxScaler(feature_range=(0, 1))
            self.X = scaler.fit_transform(self.X)

    def split_train_test_5fold(self, validation=False):

        c = list(zip(np.array(self.X), np.array(self.Y)))
        random.shuffle(c)
        X_shuffled, Y_shuffled = zip(*c)

        x_TRAINS = []
        y_TRAINS = []
        x_TESTS = []
        y_TESTS = []

        validation_split_point = round(len(self.X) * self.validations_percentage) if validation else 0
        validation_X = X_shuffled[:validation_split_point]
        validation_Y = Y_shuffled[:validation_split_point]

        X_shuffled = X_shuffled[validation_split_point:]
        Y_shuffled = Y_shuffled[validation_split_point:]

        for i in range(0, 5):

            size_of_the_fold = round(len(X_shuffled)/5.0)
            split_point1 = i * size_of_the_fold
            split_point2 = split_point1 + size_of_the_fold
            x_TRAINS.append(X_shuffled[:split_point1] + X_shuffled[split_point2:])
            y_TRAINS.append(Y_shuffled[:split_point1] + Y_shuffled[split_point2:])
            x_TESTS.append(X_shuffled[split_point1:split_point2])
            y_TESTS.append(Y_shuffled[split_point1:split_point2])

        if validation:
            return np.array(validation_X), np.array(validation_Y), np.array(x_TRAINS), np.array(y_TRAINS), np.array(x_TESTS), np.array(y_TESTS)
        else:
            return np.array(x_TRAINS), np.array(y_TRAINS), np.array(x_TESTS), np.array(y_TESTS)











