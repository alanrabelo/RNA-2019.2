import numpy as np

class Perceptron():

    weights = []

    def __init__(self, epochs=200):


        self.epochs = epochs




    def fit(self, X, Y):

        weights = np.random.uniform(low=0, high=1, size=(len(X[0])+1,))

        for epoch in range(self.epochs):

            error_count = 0

            for index, input in enumerate(X):

                input = np.insert(input, 0, -1)
                u = input.dot(weights)
                y = 1 if u > 0 else 0
                d = Y[index]
                e = d - y

                if not e == 0:
                    error_count += 1

                weights = weights + (e * input)

            if error_count == 0:
                return weights

        return None

    def predict(self, input):

        input = np.insert(input, 0, -1)
        u = input.dot(self.weights)
        return 1 if u > 0 else 0

    def evaluate(self, X, Y):

        error_count = 0

        for index, input in enumerate(X):

            desired = Y[index]
            output = self.predict(input)

            if not desired == output:
                error_count += 1

        return 1 - (error_count / len(Y))


