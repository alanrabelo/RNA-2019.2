import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import random
from ELM import ELM
import collections
import numpy as np

datasets = [
    ClassificationDatasets.ARTIFICIAL_XOR,
    ClassificationDatasets.IRIS,
    ClassificationDatasets.CANCER,
    ClassificationDatasets.DERMATOLOGY,
    ClassificationDatasets.COLUNA,
]

def run():

    for dataset in datasets:
        print('INICIANDO VALIDAÇÃO PARA O %s' % dataset.name)

        hit_sum = []
        data_manager = DataManager(dataset)
        data_manager.load_data(categorical=True)

        for i in range(0, 10):
            x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                             data_manager.Y)

            for fold in range(0, 5):

                rbf = ELM(number_of_hidden=1000)
                rbf.fit(x_TRAIN[fold], y_TRAIN[fold])
                hit_sum.append(rbf.evaluate(x_VALIDATION[fold], y_VALIDATION[fold]))

        print('Accuracy: %.2f' % np.average(hit_sum))
        print('Min: %.2f' % np.min(hit_sum))
        print('Max: %.2f' % np.max(hit_sum))
        print('Stand De: %.2f%%' % np.std(hit_sum))

run()
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

