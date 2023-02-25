import numpy as np
from scipy.spatial.distance import euclidean

def firefly_algorithm(fitness_function, dim, n, alpha=0.2, beta0=1.0, gamma=1.0, theta=1.0, max_iter=100):
    """
    Implements the Firefly Algorithm for optimization problems
    """

    # initialize the firefly population
    population = np.random.uniform(size=(n, dim))

    # evaluate the fitness of the initial population
    fitness = np.apply_along_axis(fitness_function, 1, population)

    # find the index of the best firefly
    best_index = np.argmin(fitness)

    # initialize the best firefly and its fitness
    best_firefly = population[best_index]
    best_fitness = fitness[best_index]

    # run the algorithm for the specified number of iterations
    for i in range(max_iter):

        # update the light intensity of each firefly
        for j in range(n):
            for k in range(n):
                if fitness[k] < fitness[j]:
                    r = euclidean(population[j], population[k])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    population[j] += alpha * beta * (population[k] - population[j]) + \
                                     theta * np.random.normal(size=dim)
                    population[j] = np.clip(population[j], 0, 1)

        # evaluate the fitness of the updated population
        fitness = np.apply_along_axis(fitness_function, 1, population)

        # find the index of the best firefly
        best_index = np.argmin(fitness)

        # update the best firefly and its fitness
        if fitness[best_index] < best_fitness:
            best_firefly = population[best_index]
            best_fitness = fitness[best_index]

    return best_firefly