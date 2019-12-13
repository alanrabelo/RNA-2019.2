import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import random
from Models.RBF_classification import RBFClassification
import collections
import numpy as np

datasets = [
    ClassificationDatasets.ARTIFICIAL_XOR,
    ClassificationDatasets.IRIS,
    ClassificationDatasets.CANCER,
    ClassificationDatasets.DERMATOLOGY,
    ClassificationDatasets.COLUNA,
]

sigmas = range(2, 30, 2)
number_of_centroids = range(3, 50, 5)
# sigmas = range(2, 10, 2)
# number_of_centroids = range(3, 20, 5)

def cross_validation():

    for dataset in datasets:
        print('INICIANDO VALIDAÇÃO PARA O %s' % dataset.name)

        hit_sum_test = []
        data_manager = DataManager(dataset)
        data_manager.load_data(categorical=True)
        BEST_SIGMAS = []
        BEST_NCENTR = []

        for index in range(0, 20):

            chosen_centroid_numbers = []
            chosen_sigma_numbers = []
            best_accuracies = []

            x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                                               data_manager.Y)

            best_result = 0
            best_sigma = 0
            best_cent_num = 0

            for number_of_centroid in number_of_centroids:
                for sigma in sigmas:

                    x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = data_manager.split_train_test_5fold(x_TRAIN[0], y_TRAIN[0])

                    hit_sum = []

                    for fold in range(0, 5):

                        rbf = RBFClassification(number_of_centroids=number_of_centroid, sigma=sigma)
                        rbf.fit(x_TRAINS[fold], y_TRAINS[fold])
                        hit_sum.append(rbf.evaluate(x_TESTS[fold], y_TESTS[fold]))

                    average_hit = np.average(hit_sum)
                    if average_hit > best_result:
                        best_sigma = sigma
                        best_cent_num = number_of_centroid
                        best_result = average_hit


            chosen_centroid_numbers.append(best_cent_num)
            chosen_sigma_numbers.append(best_sigma)
            best_accuracies.append(best_result)

            counts = collections.Counter(chosen_centroid_numbers)
            most_frequent = sorted(chosen_centroid_numbers, key=lambda x: (-counts[x], x), reverse=False)
            best_centroid_number = most_frequent[0]

            counts = collections.Counter(chosen_sigma_numbers)
            most_frequent = sorted(chosen_sigma_numbers, key=lambda x: (-counts[x], x), reverse=False)
            best_sigma_value = most_frequent[0]

            validation_rbf = RBFClassification(number_of_centroids=best_centroid_number, sigma=best_sigma_value)
            validation_rbf.fit(x_TRAIN[0], y_TRAIN[0])
            hit_sum_test.append(validation_rbf.evaluate(x_VALIDATION[0], y_VALIDATION[0]))

            BEST_NCENTR.append(best_centroid_number)
            BEST_SIGMAS.append(best_sigma_value)

        counts = collections.Counter(BEST_SIGMAS)
        most_frequent = sorted(BEST_SIGMAS, key=lambda x: (-counts[x], x), reverse=False)
        BEST_SIGMA = most_frequent[0]

        counts = collections.Counter(BEST_NCENTR)
        most_frequent = sorted(BEST_NCENTR, key=lambda x: (-counts[x], x), reverse=False)
        BEST_NCENT = most_frequent[0]

        print('Accuracy: %.2f' % np.average(hit_sum_test))
        print('Min: %.2f' % np.min(hit_sum_test))
        print('Max: %.2f' % np.max(hit_sum_test))
        print('Stand De: %.2f%%' % np.std(hit_sum_test))
        print('Best sigma value: %d' % BEST_SIGMA)
        print('Best number of centroids value: %d' % BEST_NCENT)

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

cross_validation()
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

