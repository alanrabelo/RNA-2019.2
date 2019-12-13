import numpy
from deap import algorithms
import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset.datasets import ClassificationDatasets, RegressionDatasets
from data_manager import DataManager
import matplotlib
matplotlib.interactive(True)
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import random
from Models.RBF_classification import RBFClassification
import collections
import numpy as np

dataset = ClassificationDatasets.ARTIFICIAL_AND

print('INICIANDO VALIDAÇÃO PARA O %s' % dataset.name)

hit_sum = []
data_manager = DataManager(dataset)
data_manager.load_data(categorical=True)

x_TRAIN, y_TRAIN, x_VALIDATION, y_VALIDATION = data_manager.split_train_test_5fold(data_manager.X,
                                                                     data_manager.Y)

number_of_hidden = 12
number_of_classes = len(y_TRAIN[0][0])
size_x = len(x_TRAIN[0][0])

rbf = RBFClassification(number_of_centroids=number_of_hidden, sigma=0.7)
rbf.number_of_classes = number_of_classes
rbf.generate_centroids_for_input(number_of_hidden, x_TRAIN[0])
rbf.hidden_weights = np.random.normal(size=(size_x, number_of_hidden))

# rbf.fit(x_TRAIN[0], y_TRAIN[0])


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_float", random.random)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_float, number_of_hidden * number_of_classes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    rbf.weights = np.reshape(individual, (number_of_hidden, number_of_classes))
    return [rbf.evaluate(x_TRAIN[0], y_TRAIN[0])]

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=0.08)
toolbox.register("select", tools.selTournament, tournsize=3)



pop = toolbox.population(n=100)
hof = tools.HallOfFame(5)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
CXPB, MUTPB = 0.5, 0.05

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0
GEN_NUM = 200
maxes = []
# Begin the evolution
while max(fits) < 1 and g < GEN_NUM:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    maxes.append(max(fits))
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

sorted = [individual for individual,_ in sorted(zip(pop,fits), reverse=True)]

print(sorted[0])

rbf.weights = np.reshape(sorted[0], (number_of_hidden, number_of_classes))
accuracy = rbf.evaluate(x_VALIDATION[0], y_VALIDATION[0])

#
print('Accuracy: %.2f' % accuracy)
print(rbf.predict([0, 0]))
print(rbf.predict([0, 1]))
print(rbf.predict([1, 0]))
print(rbf.predict([1, 1]))

from plotter import Plotter

Plotter.plot_XOR(rbf, x_TRAIN[0], y_TRAIN[0], x_VALIDATION[0], y_VALIDATION[0], 'ELM com AG')
# print('Min: %.2f' % np.min(hit_sum))
# print('Max: %.2f' % np.max(hit_sum))
# print('Stand De: %.2f%%' % np.std(hit_sum))