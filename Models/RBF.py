import numpy as np
import random
from scipy.spatial.distance import cdist

input = np.array([
    [1, 2, 3],
    [1, 9, 3],
    [8, 2, 3],
    [1, 11, 3],
    [8, 2, 12],
])

desired = np.array([
    18, 18, 15, 13, 22
])

SIGMA = 10

class RBF:

    def __init__(self, number_of_centroids, sigma):
        self.number_of_centroids = number_of_centroids
        self.sigma = sigma

    def generate_centroids_for_input(self, n, X):
        self.centroids = random.choices(X, k=n)

    def gauss(self, x):
        return np.exp(((x ** 2) / (2 * (self.sigma ** 2))))

    def fit(self, X, y):
        self.generate_centroids_for_input(self.number_of_centroids, X)

        final_result = []
        for x in X:
            x_repeated = np.repeat([x], self.number_of_centroids, axis=0)
            result = x_repeated * self.centroids
            cent_sum = np.sum(result, axis=1) ** 0.5

            input_result = []
            for summ in cent_sum:
                input_result.append(self.gauss(summ))
            final_result.append(input_result)
        final_result = np.array(final_result)

        inversion = np.linalg.pinv(np.array(final_result).astype(np.float))
        w = inversion.dot(y)

        self.weights = w
        return w

    def evaluate(self, input, desired):

        error_sum = []

        for index, x in enumerate(input):
            error_sum.append((desired[index] - self.predict(x)) ** 2)

        return np.average(error_sum)

    def predict(self, x):

        x_repeated = np.repeat([x], self.number_of_centroids, axis=0)
        result = x_repeated * self.centroids
        result = np.sum(result, axis=1)
        result = result ** 0.5
        results = [self.gauss(value) for value in result]

        result = self.weights.transpose() * results
        activation = sum(result)

        return activation
