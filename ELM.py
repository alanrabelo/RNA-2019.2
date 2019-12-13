import numpy as np
import random
from scipy.spatial.distance import cdist

input = np.array([
    [3, 3, 3],
    [1, 3, 7],
    [8, 18, 2],
    [1, 2, 8],
    [8, 18, 1],
])

desired = np.array([
    5, 4, 15, 3, 22
])

class ELM:

    def __init__(self, number_of_hidden):
        self.number_of_hidden = number_of_hidden

    def expo(self, x):
        return 1 / (1+ np.exp(-x))

    def input_to_hidden(self, x):
        a = np.dot(x, self.hidden_weights)
        np.insert(a, 0, -1)
        return a

    def fit(self, x_train, y_train):
        size_x = len(x_train[0])
        self.number_of_classes = len(y_train.shape)

        self.hidden_weights = np.random.normal(size=(size_x + 1, self.number_of_hidden))

        X = [np.insert(input, 0, -1) for input in x_train]
        X = self.input_to_hidden(X)

        Xt = np.transpose(X)
        self.output_weights = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y_train))

    def predict(self, x):
        x = self.input_to_hidden(np.insert(x, 0, -1))
        y = np.dot(x, self.output_weights)
        return y

    def predict_categorical(self, x):
        x = self.input_to_hidden(np.insert(x, 0, -1))
        y = np.dot(x, self.output_weights)
        output_categorical = np.zeros(len(self.output_weights[0]))
        output_categorical[np.argmax(y)] = 1
        return y

    def evaluate(self, input, desired):

        if self.number_of_classes == 1:
            error_sum = []
            for index, x in enumerate(input):
                d = desired[index]
                output = self.predict(x)
                error_sum.append((d - output) ** 2)

            return np.average(error_sum)
        else:
            error_count = 0
            for index, x in enumerate(input):
                d = desired[index]
                output = self.predict_categorical(x)
                output_categorical = np.zeros(len(desired[0]))
                output_categorical[np.argmax(output)] = 1
                error_count += 1 if sum((d - output_categorical) ** 2) > 0 else 0

            return 1 - error_count/len(input)
