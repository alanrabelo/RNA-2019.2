import random
import numpy as np
from mpl_toolkits import mplot3d

from data_manager import *
from Models.adaline import *
from Dataset.datasets import Datasets
import matplotlib.pyplot as plt

datasets = [Datasets.SETOSA, Datasets.VIRGINICA, Datasets.VERSICOLOR, Datasets.ARTIFICIAL]
ELEMENTOS = 100

artificial_1 = []
results = []

def f1(x):
    return 5 * x[0] + 3

def f2(x):
    return 5*x[0] - 2*x[1] + 3


def generate_f2():

    input = []
    output = []

    for i in range(0, 10):
        for j in range(0, 10):
            x = [i/100.0, j/100.0]
            input.append(x)

            noise = random.uniform(-1, 1)
            result = f2(x) + noise * 2
            output.append(result)

    return np.array(input), np.array(output)


for elemento in range(ELEMENTOS):

    noise = random.uniform(-1, 1)
    x = [elemento/100]
    result = f1(x) + noise*0.5
    artificial_1.append(x)
    results.append(result)


# for index in range(0, 1):
#
#     s = np.arange(0, len(artificial_1), 1)
#
#     np.random.shuffle(s)
#     x_data = np.array(artificial_1)[s]
#     x_label = np.array(results)[s]
#     split_point = round(len(s)*0.8)
#     train_indexes = s[:split_point]
#     test_indexes = s[split_point:]
#
#     x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = (x_data[train_indexes], x_label[train_indexes], x_data
#     [test_indexes], x_label[test_indexes])
#
#     plt.plot(x_TRAINS, y_TRAINS, 'ro')
#     plt.plot(x_TESTS, y_TESTS, 'bo')
#     plt.ylabel('some numbers')
#     plt.show()
#
#     adaline = Adaline()
#     result = adaline.fit(x_TRAINS, y_TRAINS, 'Caso 1')
#     adaline.weights = result
#
#     adaline.plot_decision_surface(x_TRAINS, y_TRAINS, x_TESTS, y_TESTS)
#
#     results.append(adaline.evaluate(x_TESTS, y_TESTS))
#
# print('MSE: %.2f' % np.average(results))
# print('RMSE: %.2f' % np.average(np.array(results)**0.5))
# print('Stand De: %.2f%%' % np.std(results))


input, output = generate_f2()
ax = plt.axes(projection="3d")

z_points = output
x_points = input[:, 0]
y_points = input[:, 1]
ax.scatter3D(x_points, y_points, z_points, c=z_points)
# plt.plot(output, input, 'ro')
# plt.ylabel('some numbers')
plt.show()