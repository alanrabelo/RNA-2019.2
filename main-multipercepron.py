from Models.adaline import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import random
from Models.MLP import MultiLayerPerceptron
import collections

datasets = [ClassificationDatasets.IRIS,
            ClassificationDatasets.CANCER,
            ClassificationDatasets.DERMATOLOGY,
            ClassificationDatasets.COLUNA,
            ClassificationDatasets.ARTIFICIAL_XOR]

number_of_hidden = range(2, 15, 2)

def cross_validation():

    for dataset in datasets:
        print('INICIANDO VALIDAÇÃO PARA O %s' % dataset.value)

        hit_sum_test = []
        data_manager = DataManager(dataset)
        data_manager.load_data()
        the_best_numbers = []

        for index in range(0, 20):

            x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                                               data_manager.Y)
            chosen_numbers = []
            best_accuracies = []

            best_result = 0
            best_number_of_hidden = 0

            for hidden_number in number_of_hidden:

                x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = data_manager.split_train_test_5fold(x_TRAIN[0], y_TRAIN[0])

                hit_sum = []

                for fold in range(0, 5):

                    perceptron = MultiLayerPerceptron(activation='tanh', hidden_number=hidden_number, learning_rate=0.01)
                    perceptron.fit(x_TRAINS[fold], y_TRAINS[fold], dataset.value, error_graph=True, epochs=300)
                    hit_sum.append(perceptron.evaluate(x_TESTS[fold], y_TESTS[fold]))

                if np.average(hit_sum) > best_result:
                    best_number_of_hidden = hidden_number

            chosen_numbers.append(best_number_of_hidden)
            best_accuracies.append(best_result)

            counts = collections.Counter(chosen_numbers)
            most_frequent = sorted(chosen_numbers, key=lambda x: (-counts[x], x), reverse=False)
            the_best_numbers.append(most_frequent[0])

            validation_perceptron = MultiLayerPerceptron(activation='tanh', hidden_number=most_frequent[0], learning_rate=0.01)
            validation_perceptron.fit(x_TRAIN[0], y_TRAIN[0], dataset.value, epochs=300)
            hit_sum_test.append(validation_perceptron.evaluate(x_VALIDATION[0], y_VALIDATION[0]))

        counts = collections.Counter(the_best_numbers)
        most_frequent = sorted(the_best_numbers, key=lambda x: (-counts[x], x), reverse=False)

        print('Accuracy: %.2f' % np.average(hit_sum_test))
        print('Min: %.2f' % np.min(hit_sum_test))
        print('Max: %.2f' % np.max(hit_sum_test))
        print('Stand De: %.2f%%' % np.std(hit_sum_test))
        print('Número de Neurônios: %d' % most_frequent[0])

def decision_surface():

    for dataset in datasets:
        print('INICIANDO o plot da superfície de decisão PARA O %s' % dataset.value)

        data_manager = DataManager(dataset)
        data_manager.load_data()

        x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                                           data_manager.Y)

        validation_perceptron = MultiLayerPerceptron(activation='tanh', hidden_number=14, learning_rate=0.01)
        # validation_perceptron.fit(x_TRAIN[0], y_TRAIN[0], dataset.value, epochs=300)
        # validation_perceptron.evaluate(x_VALIDATION[0], y_VALIDATION[0], should_print_confusion_matrix=True)
        validation_perceptron.plot_decision_surface(x_TRAIN[0], y_TRAIN[0], x_VALIDATION[0], y_VALIDATION[0], dataset.value)

decision_surface()
# input, output = generate_f2()
# # ax = plt.axes(projection="3d")
#
# for index in range(0, 30):
#
#     s = np.arange(0, len(input), 1)
#
#     np.random.shuffle(s)
#     x_data = np.array(input)[s]
#     x_label = np.array(output)[s]
#     split_point = round(len(s)*0.8)
#     train_indexes = s[:split_point]
#     test_indexes = s[split_point:]
#
#     x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = (x_data[train_indexes], x_label[train_indexes], x_data
#     [test_indexes], x_label[test_indexes])
#
#     adaline = Adaline()
#     result = adaline.fit(x_TRAINS, y_TRAINS, 'Caso 2')
#     adaline.weights = result
#
#     # adaline.plot_decision_surface(x_TRAINS, y_TRAINS, x_TESTS, y_TESTS)
#
#     adaline_results.append(adaline.evaluate(x_TESTS, y_TESTS))
#
# print('MSE: %.2f' % np.average(adaline_results))
# print('RMSE: %.2f' % np.average(np.array(adaline_results)**0.5))
# print('Stand De: %.2f%%' % np.std(adaline_results))

    # z_points = output
    # x_points = input[:, 0]
    # y_points = input[:, 1]
    # z_surface = []
    #
    # for index in range(len(x_points)):
    #     z_point = adaline.predict([x_points[index], y_points[index]])
    #     z_surface.append(z_point)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # print(z_points)
    #
    # # naming the x axis
    # plt.xlabel('X')
    # # naming the y axis
    # plt.ylabel('Y')
    # ax.plot_trisurf(x_points, y_points, z_surface, linewidth=0.2, antialiased=True, cmap=plt.cm.PuRd)
    # ax.scatter(x_points, y_points, z_points, linewidth=0.2, antialiased=True, color='blue')
    # ax.scatter(x_points, y_points, z_points, linewidth=0.2, antialiased=True, color='blue')
    #
    # # plt.ylabel('some numbers')
    # plt.show()
    # plt.close()

