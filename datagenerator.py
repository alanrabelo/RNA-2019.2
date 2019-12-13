import numpy as np
import math
import matplotlib.pyplot as plt

start = 0
stop = 9
variation = 0.2
number_of_elements = 500

def function(x):
    return (3 * math.sin(x)) + 1

X = np.linspace(start, stop, num=number_of_elements)
Y = [function(x) for x in X]
noise = np.random.uniform(-variation, variation, number_of_elements)
Y_noise = Y+noise

file = open('Dataset/regression.data', 'w')

for index, value in enumerate(X):
    str_x_value = '%.2f,%.2f\n' % (value, Y_noise[index])
    file.write(str_x_value)
file.close()

plt.plot(X, Y_noise, 'ro')
plt.plot(X, Y)
plt.show()

# variation = 0.2
# number_of_elements = 500
#
# combinations = [[0, 0,0], [0,1,1], [1,0,1], [1,1,0]]
# y = []
# number_of_elements_in_class = round(500 / len(combinations))
# X = []
# for combination in combinations:
#     [X.append(x) for x in np.repeat([combination], number_of_elements_in_class, axis=0)]
#
# y = np.array(X)[:, -1]
# X = np.array(X)[:, :-1]
#
# noise1 = np.random.uniform(-variation, variation, number_of_elements)
# noise2 = np.random.uniform(-variation, variation, number_of_elements)
# noise_joint = np.concatenate(([noise1], [noise2]), axis=0).transpose()
# X = X + noise_joint
# file = open('Dataset/XOR.data', 'w')
#
# for index, value in enumerate(X):
#     str_x_value = '%.2f,%.2f,%d\n' % (value[0], value[1], y[index])
#     file.write(str_x_value)
# file.close()

# def function(x):
#     return (3 * math.sin(x)) + 1
#
# X = np.linspace(start, stop, num=number_of_elements)
# Y = [function(x) for x in X]
# noise = np.random.uniform(-variation, variation, number_of_elements)
# Y_noise = Y+noise





