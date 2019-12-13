from enum import Enum

class ClassificationDatasets(Enum):
    IRIS = 'iris.data'
    COLUNA = 'column_3C.data'
    DERMATOLOGY = 'dermatology.data'
    CANCER = 'breast-cancer-wisconsin.data'
    ARTIFICIAL_XOR = 'XOR.data'
    ARTIFICIAL_AND = 'AND.data'
    ARTIFICIAL_OR = 'OR.data'

class RegressionDatasets(Enum):
    MOTOR = 'pmsm_temperature_data.csv'
    FUEL = 'measurements.csv'
    ABALONE = 'abalone.csv' # número de ríngs - última coluna
    ARTIFICIAL = 'regression.data'

