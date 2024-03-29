import numpy as np
from data_manager import DataManager
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn-whitegrid')

class Adaline():

    def __init__(self, epochs=5000, learning_rate=0.01):
        self.epochs = epochs
        self.weights = []
        self.learning_rate = learning_rate

    def fit(self, x, Y, dataset):

        self.x = x
        self.y = Y
        weights = np.zeros(len(x[0]) + 1)
        errors_in_epochs = []
        for epoch in range(self.epochs):

            error_count = 0
            learning_rate = self.step_decay(epoch)
            for index, input in enumerate(x):

                input = np.insert(input, 0, -1)

                u = input.dot(weights)

                y = u
                d = Y[index]
                e = d - y

                error_count += e
                weights = weights + (e * input * learning_rate)

            if epoch > 0 and error_count**0.5 < 0.2:
                errors_in_epochs.append(1)
                self.plot_error_graph(errors_in_epochs, dataset)
                return weights

            errors_in_epochs.append(error_count**0.5)

        self.plot_error_graph(errors_in_epochs, dataset)
        return weights

    def step_decay(self, epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def predict(self, data_input):

        data_input = np.insert(data_input, 0, -1)
        u = data_input.dot(self.weights)
        return u

    def evaluate(self, X, Y):

        self.X_test = X
        self.Y_test = Y
        error_sum = 0

        for index, input in enumerate(X):

            desired = Y[index]
            output = self.predict(input)
            error_sum += (desired - output)**2

        return error_sum


    def plot_error_graph(self, errors, dataset):
        x = np.linspace(0, len(errors), len(errors))
        plt.plot(x, errors)


        # naming the x axis
        plt.xlabel('Número de épocas')
        # naming the y axis
        plt.ylabel('RMSE')

        plt.suptitle(dataset)

        # giving a title to my graph
        # function to show the plot
        plt.savefig(dataset+'.png', bbox_inches='tight')
        plt.close()


    def plot_decision_surface(self, x_TRAINS, y_TRAINS, x_TESTS, y_TESTS, title='Título'):

        x = np.linspace(0, len(x_TRAINS), len(x_TRAINS))

        new_input = []

        for input_line in self.x:

            new_value = [-1]
            for value in input_line:
                new_value.append(value)

            new_input.append(new_value)

        # results = np.dot(self.weights, np.transpose(new_input))

        plt.plot(x, y_TRAINS, 'ro')

        # naming the x axis
        plt.xlabel('Número de épocas')
        # naming the y axis
        plt.ylabel('RMSE')

        plt.suptitle(title)

        # giving a title to my graph
        # function to show the plot
        # plt.savefig(dataset + '.png', bbox_inches='tight')
        plt.show()
        plt.close()


