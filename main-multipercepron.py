from Models.adaline import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset.datasets import Datasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import random
from Models.multi_perceptron import MultiPerceptron
#
# c1 = [0.5, 5.0]
# c2 = [3.5, 1.0]
# c3 = [6.5, 5.0]
#
# dataset = []
#
# file = open('Dataset/artificial_3C.data', 'w')
#
# colors = ['r', 'g', 'b']
# for index, c in enumerate([c1, c2, c3]):
#     datinha = []
#     for j in range(0, 50):
#         random1 = random.uniform(-1, 1)
#         random2 = random.uniform(-1, 1)
#         new_value = c.copy()
#         new_value[0] += random1
#         new_value[1] += random2
#         new_value.append(index)
#         dataset.append(new_value)
#         file.write('%s,%s,%s\n' % (new_value[0], new_value[1], index))
#
# file.close()
        # datinha.append(new_value)
    # datinha = np.array(datinha)
    # plt.plot(datinha[:, 0], datinha[:, 1], colors[index]+'o')

# plt.show()
#
# print(dataset)
datasets = [Datasets.IRIS, Datasets.CANCER,  Datasets.DERMATOLOGY, Datasets.COLUNA]
datasets = [Datasets.ARTIFICIAL]
datasets = [Datasets.ARTIFICIAL, Datasets.IRIS]

for dataset in datasets:

    perceptron_results = []

    for index in range(0, 1):

        print(index)
        data_manager = DataManager(dataset)
        x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = data_manager.split_train_test_5fold()

        perceptron = MultiPerceptron(activation='sigmoidal')

        for fold in range(0, 1):

            # result = perceptron.fit(x_TRAINS[fold], y_TRAINS[fold], dataset.value, error_graph=True)
            # perceptron.weights = result
            # perceptron_results.append(perceptron.evaluate(x_TESTS[fold], y_TESTS[fold]))
            perceptron.plot_decision_surface(x_TRAINS[fold], x_TESTS[fold], y_TRAINS[fold], y_TESTS[fold], name=dataset)

    print(dataset)
    print('Accuracy: %.2f' % np.average(perceptron_results))
    print('Stand De: %.2f%%' % np.std(perceptron_results))


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

