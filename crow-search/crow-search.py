import numpy as np

# Define the problem to be optimized
def objective_function(x):
    return np.sum(np.square(x))

# Define the Crow Search Algorithm function
def crow_search_algorithm(objective_function, lb, ub, max_iter, num_crows):
    # Initialize the population of crows randomly in the search space
    crows = np.random.uniform(lb, ub, (num_crows, len(lb)))

    # Initialize the best position and fitness
    best_position = np.zeros(len(lb))
    best_fitness = np.inf

    # Evaluate the fitness of each crow in the population
    fitness = np.zeros(num_crows)
    for i in range(num_crows):
        fitness[i] = objective_function(crows[i])
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_position = crows[i].copy()

    # Iterate until a termination criterion is met
    for t in range(max_iter):
        # Generate a new population of crows
        new_crows = np.zeros((num_crows, len(lb)))
        for i in range(num_crows):
            # Determine the best crow in the population
            best_crow = np.argmin(fitness)
            
            # Generate a new position for the current crow using the best crow
            new_position = np.zeros(len(lb))
            for j in range(len(lb)):
                a = np.random.uniform(-1, 1)
                b = np.random.uniform(-1, 1)
                new_position[j] = crows[i][j] + a * (crows[i][j] - crows[best_crow][j]) + b * (best_position[j] - crows[i][j])
                if new_position[j] < lb[j]:
                    new_position[j] = lb[j]
                elif new_position[j] > ub[j]:
                    new_position[j] = ub[j]
            
            # Evaluate the fitness of the new position
            new_fitness = objective_function(new_position)
            
            # Update the best position and fitness
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_position = new_position.copy()
            
            # Add the new crow to the new population
            new_crows[i] = new_position

        # Replace the worst crows in the current population with the best crows from the new population
        for i in range(num_crows):
            worst_crow = np.argmax(fitness)
            if fitness[i] > fitness[worst_crow]:
                crows[worst_crow] = new_crows[i]
                fitness[worst_crow] = fitness[i]

    return best_position, best_fitness