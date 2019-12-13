import numpy as np
import random
from scipy.spatial.distance import cdist

class RBFClassification:

    def __init__(self, number_of_centroids, sigma):
        self.number_of_centroids = number_of_centroids
        self.sigma = sigma
        self.weights = 0

    def generate_centroids_for_input(self, n, X):
        self.centroids = random.choices(X, k=n)

    def gauss(self, x):
        return np.exp(((x ** 2) / (2 * (self.sigma ** 2))))

    def fit(self, X, y):

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
            y = self.predict(x)
            d = desired[index]
            error_count += 1 if sum((d - y) ** 2) > 0 else 0

        return 1 - (error_count / len(input))

    def predict(self, x):

        result = np.dot([x], np.array(self.centroids).transpose())
        # result = result ** 0.5
        results = [self.gauss(value) for value in result]
        result_final = np.dot(results, self.weights)

        response = np.zeros(len(result_final[0]))
        response[np.argmax(result_final)] = 1
        return response
