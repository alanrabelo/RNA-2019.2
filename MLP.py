from data_manager import *
from Models.MLP import *
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
import pickle as pkl
datasets = [ClassificationDatasets.IRIS,
ClassificationDatasets.COLUNA,
ClassificationDatasets.DERMATOLOGY,
ClassificationDatasets.CANCER,
ClassificationDatasets.ARTIFICIAL_XOR]
datasets = [ClassificationDatasets.IRIS, ClassificationDatasets.ARTIFICIAL_XOR]
# datasets = [Datasets.IRIS, Datasets.ARTIFICIAL_XOR]
NUMBER_OF_FOLDS = 5
NUMBER_OF_REALIZATIONS = 5

NEURONS_RANGE = range(3, 15, 2)
NEURONS_RANGE = [5, 12]

for dataset in datasets:

    print('Treinando para o %s' % dataset.value)
    results = {}

    number_of_hidden_neurons = NEURONS_RANGE
    manager = DataManager(dataset)
    x_TRAINNING, y_TRAINNING, x_VALIDATION, y_VALIDATION = manager.split_train_test_5fold(manager.X, manager.Y, hasvalidation=False)

    for number_of_neurons in number_of_hidden_neurons:

        correctness_sum = 0

        for index in range(0, NUMBER_OF_REALIZATIONS):

            x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = manager.split_train_test_5fold(x_TRAINNING, y_TRAINNING, hasvalidation=True)

            for i in range(0, NUMBER_OF_FOLDS):

                perceptron = MultiLayerPerceptron(hidden_number=number_of_neurons)
                result = perceptron.fit(x_TRAINS[i], y_TRAINS[i], dataset.value, error_graph=True, epochs=500)
                perceptron.weights, perceptron.hidden_weights = result

                correctness = perceptron.evaluate(x_TESTS[i], y_TESTS[i], should_print_confusion_matrix=True if index == 4 and i == 0 else False)
                correctness_sum += correctness
                if index == 0:
                    perceptron.plot_decision_surface(x_TRAINS[i], y_TRAINS[i], x_TESTS[i], y_TESTS[i], 'MLP - '+dataset.value)

        results[str(number_of_neurons)] = correctness_sum/(NUMBER_OF_FOLDS*NUMBER_OF_REALIZATIONS)

    best_combination = sorted(results.items(), key=lambda x: x[0], reverse=True)[0]
    print(str(best_combination) + ' best combination for ' + dataset.value)
    perceptron = MultiLayerPerceptron(hidden_number=int(best_combination[0]))
    results = perceptron.fit(x_TRAINNING, y_TRAINNING, dataset.value, error_graph=True, epochs=500)
    perceptron.weights, perceptron.hidden_weights = results

    result = perceptron.evaluate(x_VALIDATION, y_VALIDATION, should_print_confusion_matrix=True)
    print('Result for ' + dataset.value + ': ' + str(result))

    with open('Results for ' + dataset.value + '.pkl', 'wb') as f:
        pkl.dump(results, f)

    # print('Accuracy: %.2f%%' % (np.average(results)))
    # print('Stand De: %.2f%%' % np.std(results))

# with open('Results for ' + dataset.value + '.pkl', 'r') as f:
#     results_loaded = pkl.load(f)
#     print(results_loaded)