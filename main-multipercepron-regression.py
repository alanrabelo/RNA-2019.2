from Models.adaline import *
from Dataset.datasets import RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
from Models.MLP_regression import MultiLayerPerceptronRegressor

datasets = [
    RegressionDatasets.ARTIFICIAL,
    RegressionDatasets.FUEL,
    RegressionDatasets.ABALONE,
    RegressionDatasets.MOTOR,
]

def validate():

    for dataset in datasets:
        print('INICIANDO o plot da superfície de decisão PARA O %s' % dataset.name)

        data_manager = DataManager(dataset)
        data_manager.load_data(categorical=False)
        MSE = []
        RMSE = []

        for realization in range(0, 10):

            x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                                               data_manager.Y)
            for i in range(0, 1):

                validation_perceptron = MultiLayerPerceptronRegressor(activation='tanh', hidden_number=15, learning_rate=0.01)
                validation_perceptron.fit(x_TRAIN[i], y_TRAIN[i], dataset.value, epochs=300, verbose=True)
                mse, rmse = validation_perceptron.evaluate(x_VALIDATION[i], y_VALIDATION[i])
                MSE.append(mse)
                RMSE.append(rmse)

        print('MSE: %.2f' % np.average(MSE))
        print('RMSE: %.2f' % np.average(RMSE))
        print('Stand De MSE: %.2f%%' % np.std(MSE))
        print('Stand De RMSE: %.2f%%' % np.std(RMSE))

# decision_surface()
validate()