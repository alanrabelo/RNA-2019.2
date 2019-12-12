import numpy as np
import math
import matplotlib.pyplot as plt

start = 0
stop = 25
variation = 0.5
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





