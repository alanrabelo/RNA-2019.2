import numpy as np
from data_manager import DataManager
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class Perceptron():

    def __init__(self, epochs=200, learning_rate=0.01):
        self.epochs = epochs
        self.weights = []
        self.learning_rate = learning_rate

    def fit(self, x, Y, dataset, error_graph=True):

        weights = np.random.uniform(low=0, high=1, size=(len(x[0]) + 1,))
        errors_in_epochs = []
        for epoch in range(self.epochs):

            error_count = 0

            for index, input in enumerate(x):

                input = np.insert(input, 0, -1)
                u = input.dot(weights)
                y = 1 if u > 0 else 0
                d = Y[index]
                e = d - y

                if not e == 0:
                    error_count += 1

                weights = weights + (e * input * self.learning_rate)

            if error_count == 0:
                errors_in_epochs.append(1)
                self.plot_error_graph(errors_in_epochs, dataset)
                return weights

            errors_in_epochs.append(1 - (error_count/len(x)))

        self.plot_error_graph(errors_in_epochs, dataset)
        return weights

    def predict(self, data_input):

        data_input = np.insert(data_input, 0, -1)
        u = data_input.dot(self.weights)
        return 1 if u > 0 else 0

    def evaluate(self, X, Y, should_print_confusion_matrix=False):

        error_count = 0
        confusion_matrix = {}

        for index, input in enumerate(X):

            desired = Y[index]
            output = self.predict(input)

            if not desired == output:
                error_count += 1

            if desired in confusion_matrix:
                if output in confusion_matrix[desired]:
                    confusion_matrix[desired][output] += 1
                else:
                    confusion_matrix[desired][output] = 1
            else:
                confusion_matrix[desired] = {output: 1}

        if should_print_confusion_matrix:
            print('Matriz de confusão: \n%s' % confusion_matrix)

        return 1 - (error_count / len(Y))


    def plot_error_graph(self, errors, dataset):
        x = np.linspace(0, len(errors), len(errors))
        plt.plot(x, errors)


        # naming the x axis
        plt.xlabel('Número de épocas')
        # naming the y axis
        plt.ylabel('Taxa de acerto [0 - 1]')

        plt.suptitle(dataset)

        # giving a title to my graph
        # function to show the plot
        plt.savefig(dataset+'.png', bbox_inches='tight')
        plt.close()


    def plot_decision_surface(self, filename, title, folder_to_save='Plots/', same_covariances=False):

        # parameter_combination = list(itertools.combinations(range(len(list(X[0]))), 2))

        manager = DataManager(filename)


        x_train, y_train, x_test, y_test = manager.split_train_test_5fold(validation=False)

        x_train = x_train[0][:, :2]
        x_test = x_test[0][:, :2]
        self.weights = self.fit(x_train, y_train[0])


        clear_red = "#ffcccc"
        clear_blue = "#ccffff"
        clear_green = "#ccffcc"
        clear_yellow = "#F5FAA9"
        clear_pink = "#F9A9FA"
        clear_orange = "#FBECD1"

        colors = [clear_red, clear_blue, clear_green, clear_yellow, clear_pink, clear_orange]
        strong_colors = ['red', 'blue', '#2ECC71', '#F9FF2D', '#FF2DF2', '#FFAE00']
        number_of_points = 80

        points_for_class = {}

        for i in range(0, number_of_points + 1, 1):
            for j in range(0, number_of_points + 1, 1):
                x = i / number_of_points
                y = j / number_of_points
                value = int(self.predict(np.array([x, y])))
                if value in points_for_class:
                    points_for_class[value].append([x, y])
                else:
                    points_for_class[value] = [[x, y]]

        for key in points_for_class.keys():
            points = np.array(points_for_class[key])
            plt.plot(points[:, 0], points[:, 1], 'ro', color=colors[key])

        medium_colors = ['#641E16', '#1B4F72', '#186A3B', '#AAB203', '#970297', '#974F02']

        for index, input in enumerate(x_train):
            color_value = int(y_train[0][index])
            plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])

        for index, input in enumerate(x_test):
            color_value = int(y_test[0][index])
            plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])

        plt.suptitle(title)
        plt.savefig(folder_to_save + title + '.png')


