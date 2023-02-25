import numpy as np

# Define the fitness function
def fitness_func(x):
    return np.sum(x)

# Define the local search method
def local_search(x):
    y = x.copy()
    for i in range(len(x)):
        y[i] = 1 - y[i]
        if fitness_func(y) > fitness_func(x):
            x = y.copy()
        else:
            y[i] = 1 - y[i]
    return x

# Define the memetic algorithm
def memetic_algorithm(pop_size, gene_length, mutation_rate, generations):
    # Initialize the population
    population = np.random.randint(0, 2, size=(pop_size, gene_length))

    # Iterate over the generations
    for i in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness = np.apply_along_axis(fitness_func, 1, population)

        # Select the parents for the next generation
        parents = population[np.argsort(-fitness)][:2]

        # Perform crossover to create the offspring
        offspring = np.empty((pop_size, gene_length))
        for j in range(pop_size):
            parent1 = parents[j % 2]
            parent2 = parents[(j + 1) % 2]
            mask = np.random.randint(0, 2, size=gene_length, dtype=bool)
            offspring[j][mask] = parent1[mask]
            offspring[j][~mask] = parent2[~mask]

        # Perform mutation on the offspring
        mask = np.random.random((pop_size, gene_length)) < mutation_rate
        offspring[mask] = 1 - offspring[mask]

        # Apply local search to the offspring
        offspring = np.apply_along_axis(local_search, 1, offspring)

        # Update the population with the new generation
        population = offspring

    # Return the best individual
    fitness = np.apply_along_axis(fitness_func, 1, population)
    best_individual = population[np.argmax(fitness)]

    return best_individual, np.max(fitness)

# Example usage
best_individual, max_fitness = memetic_algorithm(pop_size=10, gene_length=5, mutation_rate=0.1, generations=100)
print("Best individual: ", best_individual)
print("Max fitness: ", max_fitness)