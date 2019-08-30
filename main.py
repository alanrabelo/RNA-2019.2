from data_manager import *
from Models.simple_perceptron import *
from Dataset.datasets import Datasets

manager = DataManager(Datasets.IRIS.value)

results = []
for i in range(0, 100):
    x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = manager.split_train_test_5fold(validation=False)
    error_average = 0

    for i in range(0, 5):

        perceptron = Perceptron()
        result = perceptron.fit(x_TRAINS[i], y_TRAINS[i])
        perceptron.weights = result

        error_average += perceptron.evaluate(x_TESTS[i], y_TESTS[i])
    results.append(error_average/5.0)

print('Accuracy: %.2f%%' % np.average(results))
print('Stand De: %.2f%%' % np.std(results))
