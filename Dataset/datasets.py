from enum import Enum

class ClassificationDatasets(Enum):
    IRIS = 'iris.data'
    COLUNA = 'column_3C.data'
    DERMATOLOGY = 'dermatology.data'
    CANCER = 'breast-cancer-wisconsin.data'
    ARTIFICIAL_XOR = 'artificial_XOR.data'

class RegressionDatasets(Enum):
    MOTOR = 'pmsm_temperatura_data.csv'
    FUEL = 'measurements.csv'
    ABALONE = 'abalone.csv'
