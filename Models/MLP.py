import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import math


class MultiLayerPerceptron():

    def __init__(self, epochs=200, learning_rate=0.5, activation='sigmoid', hidden_number=10):
        self.epochs = epochs
        self.weights = []
        self.hidden_weights = []
        self.learning_rate = learning_rate
        self.activation = activation
        self.hidden_number = hidden_number


    def sigmoid(self, u):

        array = np.array(u)

        if self.activation == 'logistic':
            y = 1.0 / (1.0 + np.exp(-array))
        elif self.activation == 'tanh':
            y = np.tanh(array)
        else:
            raise ValueError('Error in function!')
        return y

    def sigmoid_(self, u):

        if self.activation == 'logistic':
            y_ = u * (1.0 - u)
        elif self.activation == 'tanh':
            y_ = 1.0 - (u * u)
        else:
            raise ValueError('Error in derivate!')
        return y_

    def updateEta(self, epoch):
        eta_i = 0.1
        eta_f = eta_i * 0.5
        eta = eta_i * ((eta_f / eta_i) ** (epoch / self.epochs))
        self.learning_rate = eta

    def fit(self, x, Y, dataset='', error_graph=True, epochs=300):

        self.number_of_classes = len(set([str(output) for output in Y]))
        x_shape = np.shape(x)[1]

        hidden_weights = np.random.random(size=(x_shape + 1, self.hidden_number))
        hidden_weights -= 0.5
        hidden_weights *= 2
        weights = np.random.uniform(size=(self.hidden_number+1, self.number_of_classes))
        weights -= 0.5
        weights *= 2

        errors_in_epochs = []
        for epoch in range(epochs):
            self.updateEta(epoch)
            error_count = 0

            for index, input_x in enumerate(x):

                # FORWARD STEP
                input_x = np.insert(input_x, 0, -1)
                x_transp = np.array([input_x])
                uh = list(input_x.dot(hidden_weights))

                h = self.sigmoid(uh)
                h_ = self.sigmoid_(h)
                h = np.insert(h, 0, -1)
                h_ = np.insert(h_, 0, -1)

                uy = h.dot(weights)
                y = self.sigmoid(uy)
                y_ = self.sigmoid_(y)

                d = Y[index]
                e = d - y

                output = list(np.zeros(self.number_of_classes, dtype=int))
                output[list(y).index(max(y))] = 1

                error = d - output
                if sum(error**2) > 0:
                    error_count += 1

                # BACKPROPAGATION OUTPUT
                h_transp = np.array([h]).transpose()
                update = np.dot(h_transp, [e * y_])
                weights += update * self.learning_rate

                # BACKPROPAGATION HIDDEN
                h__transp = np.array([h_]).transpose()

                factor1 = np.array([e * y_]).transpose()
                e_hidden = np.array(weights).dot(factor1)

                update_hidden = h__transp[1:] * e_hidden[1:]
                update_hidden = update_hidden.dot(x_transp).transpose()

                hidden_weights += self.learning_rate * update_hidden
            errors_in_epochs.append(1 - (error_count/len(x)))

        if error_graph:
            self.plot_error_graph(errors_in_epochs, dataset)

        self.weights = weights
        self.hidden_weights = hidden_weights

    def predict(self, data_input):

        data_input = np.insert(data_input, 0, -1)
        uh = data_input.dot(self.hidden_weights)
        h = self.sigmoid(uh)
        h = np.insert(h, 0, -1)

        u = np.array(self.weights).transpose().dot(h)

        y_sig = list(self.sigmoid(u))
        y = list(np.zeros(self.number_of_classes, dtype=int))
        y[y_sig.index(max(y_sig))] = 1

        return y

    def evaluate(self, X, Y, should_print_confusion_matrix=False):

        error_count = 0
        confusion_matrix = {}

        for index, input in enumerate(X):

            desired = Y[index]
            output = self.predict(input)
            error = desired - output

            if not sum(error**2) == 0:
                error_count += 1

            desired_str = str(list(desired))
            output_str = str(output)

            if desired_str in confusion_matrix:
                if output_str in confusion_matrix[desired_str]:
                    confusion_matrix[desired_str][output_str] += 1
                else:
                    confusion_matrix[desired_str][output_str] = 1
            else:
                confusion_matrix[desired_str] = {output_str: 1}

        if should_print_confusion_matrix:
            print('Matriz de confusão: \n%s' % confusion_matrix)

        return (1 - (error_count / len(Y))) * 100

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
        plt.savefig('Plots/Errors/Errors' + dataset+'.png', bbox_inches='tight')
        plt.close()


    def plot_decision_surface(self, x_train, y_train, x_test, y_test, name):

        x_train = np.array(x_train)[:, :2]
        x_test = np.array(x_test)[:, :2]
        self.fit(x_train, y_train, 'surface', error_graph=True)

        clear_red = "#ffcccc"
        clear_blue = "#ccffff"
        clear_green = "#ccffcc"
        clear_yellow = "#F5FAA9"
        clear_pink = "#F9A9FA"
        clear_orange = "#FBECD1"

        colors = [clear_red, clear_blue, clear_green, clear_yellow, clear_pink, clear_orange]
        strong_colors = ['red', 'blue', '#2ECC71', '#F9FF2D', '#FF2DF2', '#FFAE00']
        number_of_points = 100

        points_for_class = {}

        for i in range(0, number_of_points + 1, 1):
            for j in range(0, number_of_points + 1, 1):
                x = i / number_of_points
                y = j / number_of_points

                value = self.predict(np.array([x, y]))

                real_value = 0
                for index, output in enumerate(value):
                    if int(output) == 1:
                        real_value = index
                        break

                if real_value in points_for_class:
                    points_for_class[real_value].append([x, y])
                else:
                    points_for_class[real_value] = [[x, y]]

        keys = list(points_for_class.keys())
        keys = sorted(keys)

        for index, key in enumerate(keys):
            points = np.array(points_for_class[key])
            plt.plot(points[:, 0], points[:, 1], 'ro', color=colors[index])

        medium_colors = ['#641E16', '#1B4F72', '#186A3B', '#AAB203', '#970297', '#974F02']

        for index, input in enumerate(x_train):

            real_value = 0
            for index, value in enumerate(y_train[index]):
                if int(value) == 1:
                    real_value = index
                    break
            color_value = real_value
            plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])

        for index, input in enumerate(x_test):
            real_value = 0
            for index, value in enumerate(y_test[index]):
                if int(value) == 1:
                    real_value = index
                    break
            color_value = real_value
            plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])

        plt.title(name)
        # plt.show()
        plt.savefig('Plots/' + name + '.png')


