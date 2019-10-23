from data_manager import *
from Models.MLP import *
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
import pickle as pkl
import tensorflow as tf
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

datasets = [RegressionDatasets.ABALONE,
RegressionDatasets.FUEL,
]

NUMBER_OF_FOLDS = 5
NUMBER_OF_REALIZATIONS = 20
NUMBER_OF_EPOCHS = 500
NEURONS_RANGE = [3, 5, 7, 9, 12, 15]

for dataset in datasets:

    print('Treinando para o %s' % dataset.value)
    validation_accuracy = []
    best_neurons = []
    number_of_hidden_neurons = NEURONS_RANGE
    manager = DataManager(dataset)
    x_TRAINNING, y_TRAINNING, x_VALIDATION, y_VALIDATION = manager.split_train_test_5fold(manager.X, manager.Y, hasvalidation=False)
    for index in range(0, NUMBER_OF_REALIZATIONS):

        results = {}

        for number_of_neurons in number_of_hidden_neurons:

            x_TRAINS, y_TRAINS, x_TESTS, y_TESTS = manager.split_train_test_5fold(x_TRAINNING, y_TRAINNING, hasvalidation=True)
            accuracies = []

            for i in range(0, NUMBER_OF_FOLDS):

                model = Sequential()
                model.add(Dense(number_of_neurons, input_dim=len(x_TRAINS[0][0])))
                model.add(Activation('sigmoid'))
                model.add(Dense(len(y_TRAINS[0][0])))
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                model.fit(x_TRAINS[i], y_TRAINS[i], epochs=200, verbose=0)
                result = model.evaluate(x_TESTS[i], y_TESTS[i])
                # correctness = perceptron.evaluate(x_TESTS[i], y_TESTS[i], should_print_confusion_matrix=True if index == 0 and i == 0 else False)
                accuracies.append(result[1])
                # if index == 0 and i == 0:
                #     perceptron.plot_decision_surface(x_TRAINS[i], y_TRAINS[i], x_TESTS[i], y_TESTS[i], 'MLP - '+dataset.value)

            results[str(number_of_neurons)] = np.average(accuracies)

        best_combination = sorted(results.items(), key=lambda x: x[0], reverse=False)[0]
        model = Sequential()
        model.add(Dense(int(best_combination[0]), input_dim=len(x_TRAINNING[0])))
        model.add(Activation('sigmoid'))
        model.add(Dense(len(y_TRAINNING[0])))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_TRAINNING, y_TRAINNING, epochs=200)

        result = model.evaluate(x_VALIDATION, y_VALIDATION)[1]
        validation_accuracy.append(result)
        best_neurons.append(int(best_combination[0]))

    print('Result for ' + dataset.value + ': ' + str(np.average(validation_accuracy) * 100))
    print('STD was ' + dataset.value + ': ' + str(np.std(validation_accuracy)))
    counts = np.bincount(best_neurons)
    print('The best number of neurons was ' + str(np.argmax(counts)))