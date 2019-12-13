import numpy as np
import matplotlib.pyplot as plt
import math
from data_manager import DataManager
from Models.MLP import MultiLayerPerceptron
from Models.MLP_regression import MultiLayerPerceptronRegressor
from Models.RBF_classification import RBFClassification
from Models.RBF import RBF
from ELM import ELM
from Dataset.datasets import ClassificationDatasets, RegressionDatasets

class Plotter:
    # @staticmethod
    # def plot(model, x_train, y_train, x_test, y_test, name):
    #
    #     x_train = np.array(x_train)[:, :2]
    #     x_test = np.array(x_test)[:, :2]
    #     model.fit(x_train, y_train)
    #
    #     clear_red = "#ffcccc"
    #     clear_blue = "#ccffff"
    #     clear_green = "#ccffcc"
    #     clear_yellow = "#F5FAA9"
    #     clear_pink = "#F9A9FA"
    #     clear_orange = "#FBECD1"
    #
    #     colors = [clear_red, clear_blue, clear_green, clear_yellow, clear_pink, clear_orange]
    #     strong_colors = ['red', 'blue', '#2ECC71', '#F9FF2D', '#FF2DF2', '#FFAE00']
    #     number_of_points = 100
    #
    #     points_for_class = {}
    #
    #     for i in range(0, number_of_points + 1, 1):
    #         for j in range(0, number_of_points + 1, 1):
    #             x = i / number_of_points
    #             y = j / number_of_points
    #
    #             value = model.predict(np.array([x, y]))
    #
    #             real_value = 0
    #             for index, output in enumerate(value):
    #                 if int(output) == 1:
    #                     real_value = index
    #                     break
    #
    #             if real_value in points_for_class:
    #                 points_for_class[real_value].append([x, y])
    #             else:
    #                 points_for_class[real_value] = [[x, y]]
    #
    #     keys = list(points_for_class.keys())
    #     keys = sorted(keys)
    #
    #     for index, key in enumerate(keys):
    #         points = np.array(points_for_class[key])
    #         plt.plot(points[:, 0], points[:, 1], 'ro', color=colors[index])
    #
    #     medium_colors = ['#641E16', '#1B4F72', '#186A3B', '#AAB203', '#970297', '#974F02']
    #
    #     for index, input in enumerate(x_train):
    #
    #         real_value = 0
    #         for index, value in enumerate(y_train[index]):
    #             if int(value) == 1:
    #                 real_value = index
    #                 break
    #         color_value = real_value
    #         plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])
    #
    #     for index, input in enumerate(x_test):
    #         real_value = 0
    #         for index, value in enumerate(y_test[index]):
    #             if int(value) == 1:
    #                 real_value = index
    #                 break
    #         color_value = real_value
    #         plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])
    #
    #     plt.title(name)
    #     # plt.show()
    #     plt.savefig('Plots/' + name + '.png')

    @staticmethod
    def plot_XOR(model, x_train, y_train, x_test, y_test, name):

        x_train = np.array(x_train)[:, :2]
        x_test = np.array(x_test)[:, :2]

        clear_red = "#ffcccc"
        clear_blue = "#ccffff"
        clear_green = "#cplot_XORcffcc"
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

                value = model.predict(np.array([x, y]))

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
        plt.show()
        # plt.savefig('Plots/' + name + '.png')

    @staticmethod
    def decision_surface(model, model_name):

        start = 0
        stop = 10
        variation = 0.3
        number_of_elements = 500

        def function(x):
            return (3 * math.sin(x)) + 1

        X = np.linspace(start, stop, num=number_of_elements)
        X = [np.array([x]) for x in X]
        Y = [function(x) for x in X]
        noise = np.random.uniform(-variation, variation, number_of_elements)
        Y_noise = Y + noise

        data_manager = DataManager('Artificial I')
        x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(X,
                                                                                           Y_noise)
        model.fit(x_TRAIN[0], y_TRAIN[0])
        predicted = [model.predict(x) for x in x_VALIDATION[0]]
        plt.plot(x_TRAIN[0], y_TRAIN[0], 'ro')
        plt.plot(X, Y)
        plt.savefig('Plots/Regression/'+name+'-original.png')
        plt.close()
        plt.plot(x_TRAIN[0], y_TRAIN[0], 'ro')
        plt.plot(X, Y)
        plt.plot(x_VALIDATION[0], predicted, 'go')
        plt.savefig('Plots/Regression/'+name+'-result.png')



#
# classif_models = [
#     # ('MLP', MultiLayerPerceptron(hidden_number=15, epochs=500, learning_rate=0.01, activation='tanh')),
#     ('ELM', RBFClassification(50, 5)),
#     # ('ELM', ELM(1000))
# ]
# regress_models = [
#     # ('MLP', MultiLayerPerceptronRegressor(hidden_number=35, epochs=600, learning_rate=0.01, activation='tanh')),
#     ('RBF', RBF(50, 5)),
#     # ('ELM', ELM(1000))
# ]
#
# datasets_class = [
#     ClassificationDatasets.IRIS,
#     ClassificationDatasets.CANCER,
#     ClassificationDatasets.DERMATOLOGY,
#     ClassificationDatasets.COLUNA,
#     ClassificationDatasets.ARTIFICIAL_XOR,
# ]
# regression_datasets= [
#     RegressionDatasets.ARTIFICIAL,
# ]
#
# for dataset in datasets_class:
#
#     data_manager = DataManager(dataset)
#     data_manager.load_data(categorical=True)
#     x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
#                                                                                            data_manager.Y)
#
#     for name, model in classif_models:
#         Plotter.plot(model, x_TRAIN[0], y_TRAIN[0], x_VALIDATION[0], y_VALIDATION[0], name + '-' + dataset.name)
#
# # for dataset in regression_datasets:
# #
# #     data_manager = DataManager(dataset)
# #     data_manager.load_data(categorical=True)
# #     x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
# #                                                                                            data_manager.Y)
# #
# #     for name, model in regress_models:
# #         Plotter.decision_surface(model, name)