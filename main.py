from data_manager import *
from Models.simple_perceptron import *
from Dataset.datasets import Datasets

datasets = [Datasets.SETOSA, Datasets.VIRGINICA, Datasets.VERSICOLOR, Datasets.ARTIFICIAL]

for dataset in datasets:

    print('Treinando para o %s' % dataset.value)
    results = []

    for index in range(0, 1):

        manager = DataManager(dataset.value)
        x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = manager.split_train_test_5fold()
        error_average = 0

        for i in range(0, 1):

            perceptron = Perceptron()
            result = perceptron.fit(x_TRAINS[i], y_TRAINS[i], dataset.value, error_graph=True)
            perceptron.weights = result

            error_average += perceptron.evaluate(x_TESTS[i], y_TESTS[i], should_print_confusion_matrix=True if index == 4 and i == 0 else False)

            if i == 4:
                perceptron.plot_decision_surface(dataset.value, dataset.value)

        results.append(error_average/5.0)

    print('Accuracy: %.2f%%' % (np.average(results)*100))
    print('Stand De: %.2f%%' % np.std(results))
