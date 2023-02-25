import numpy as np

def dandelion_optimizer(objective_function, lb, ub, dimension, max_iter):
    # Initialize the position of dandelions
    positions = np.random.uniform(low=lb, high=ub, size=(dimension, 10))
    fitness = np.zeros(10)
    best_fitness = float('inf')
    best_position = np.zeros(dimension)

    for i in range(10):
        fitness[i] = objective_function(positions[:, i])

        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_position = positions[:, i]

    for iter in range(max_iter):
        for i in range(10):
            # Mutation operator
            mutant = positions[:, i] + np.random.normal(scale=0.1, size=dimension)

            # Boundary handling
            mutant = np.maximum(mutant, lb)
            mutant = np.minimum(mutant, ub)

            # Evaluate the fitness of mutant
            mutant_fitness = objective_function(mutant)

            # Competition operator
            if mutant_fitness < fitness[i]:
                positions[:, i] = mutant
                fitness[i] = mutant_fitness

                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_position = positions[:, i]

        # Dispersion operator
        for i in range(10):
            for j in range(dimension):
                positions[j, i] = best_position[j] + np.random.normal(scale=0.1)

                # Boundary handling
                positions[j, i] = np.maximum(positions[j, i], lb[j])
                positions[j, i] = np.minimum(positions[j, i], ub[j])

    return best_position, best_fitness