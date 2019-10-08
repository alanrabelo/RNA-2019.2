from data_manager import *
from Models.MLP import *
from Dataset.datasets import Datasets
import pickle as pkl
# datasets = [Datasets.IRIS, Datasets.COLUNA, Datasets.CANCER, Datasets.ARTIFICIAL_XOR]
datasets = [Datasets.IRIS]
# datasets = [Datasets.IRIS, Datasets.ARTIFICIAL_XOR]
NUMBER_OF_FOLDS = 5
NUMBER_OF_REALIZATIONS = 1

for dataset in datasets:

    print('Treinando para o %s' % dataset.value)
    results = {}

    for index in range(0, NUMBER_OF_REALIZATIONS):

        manager = DataManager(dataset)
        x_VALIDATION, y_VALIDATION, x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = manager.split_train_test_5fold(validation=True)
        correct_average = 0
        number_of_epochs = range(100, 1200, 100)
        number_of_hidden_neurons = range(1, 20)
        for epochs in number_of_epochs:
            for number_of_neurons in number_of_hidden_neurons:
                for i in range(0, NUMBER_OF_FOLDS):

                    perceptron = MultiLayerPerceptron(hidden_number=10)
                    result = perceptron.fit(x_TRAINS[i], y_TRAINS[i], dataset.value, error_graph=True, epochs=200)
                    perceptron.weights, perceptron.hidden_weights = result

                    correctness = perceptron.evaluate(x_TESTS[i], y_TESTS[i], should_print_confusion_matrix=True if index == 4 and i == 0 else False)
                    correct_average += correctness
                    # # if i == 4:
                    #     perceptron.plot_decision_surface(x_TRAINS[i], y_TRAINS[i], x_TESTS[i], y_TESTS[i], 'MLP - '+dataset.value)


            results.append(correct_average/NUMBER_OF_FOLDS)

    with open('Results for ' + dataset.value + '.pkl', 'wb') as f:
        pkl.dump(results, f)

    print('Accuracy: %.2f%%' % (np.average(results)))
    print('Stand De: %.2f%%' % np.std(results))

with open('Results for ' + dataset.value + '.pkl', 'r') as f:
    results_loaded = pkl.load(f)
    print(results_loaded)