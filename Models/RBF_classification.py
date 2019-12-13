import numpy as np
import random
from scipy.spatial.distance import cdist

input = np.array([
    [8, 2, 3],
    [7, 3, 3],
    [8, 1, 3],
    [1, 11, 3],
    [3, 12, 12],
])

desired = np.array([
    [0, 1], [0, 1], [0, 1], [1, 0], [1, 0],
])

SIGMA = 10

class RBFClassification:

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

        error_count = 0.0

        for index, x in enumerate(input):
            error_count += 1 if sum((desired[index] - self.predict(x)) ** 2) > 0 else 0

        return 1 - (error_count / len(input))

    def predict(self, x):

        x_repeated = np.repeat([x], self.number_of_centroids, axis=0)
        result = x_repeated * self.centroids
        result = np.sum(result, axis=1)
        result = result ** 0.5
        results = [self.gauss(value) for value in result]
        results_repeated = np.repeat([results], len(self.weights[0]), axis=0)
        result_final = self.weights.transpose() * results_repeated
        activation = np.sum(result_final, axis=1)

        response = np.zeros(len(result_final))
        response[np.argmax(activation)] = 1
        return response


rbf = RBFClassification(3, 5)
rbf.fit(input, desired)

print(rbf.evaluate([[6, 2, 3], [4, 15, 12]], [[0, 1], [1, 0]]))