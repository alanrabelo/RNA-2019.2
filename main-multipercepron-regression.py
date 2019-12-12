from Models.adaline import *
from Dataset.datasets import RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
from Models.MLP_regression import MultiLayerPerceptronRegressor

datasets = [
    # RegressionDatasets.FUEL,
    RegressionDatasets.MOTOR,
    # RegressionDatasets.ABALONE,
    # RegressionDatasets.ARTIFICIAL
        ]

def decision_surface():

    for dataset in datasets:
        print('INICIANDO o plot da superfície de decisão PARA O %s' % dataset.value)

        data_manager = DataManager(dataset)
        data_manager.load_data(categorical=False)
        MSE = []
        RMSE = []

        for i in range(0, 1):

            x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                                               data_manager.Y)

            validation_perceptron = MultiLayerPerceptronRegressor(activation='tanh', hidden_number=25, learning_rate=0.01)
            validation_perceptron.fit(x_TRAIN[0], y_TRAIN[0], dataset.value, epochs=20)
            mse, rmse = validation_perceptron.evaluate(x_VALIDATION[0], y_VALIDATION[0])
            MSE.append(mse)
            RMSE.append(rmse)

        print('MSE: %.2f' % np.average(MSE))
        print('RMSE: %.2f' % np.average(RMSE))

decision_surface()
