import numpy as np
import math

# Define the objective function
def obj_func(x):
    return sum(x**2)

# Define the Red Deer Algorithm
def red_deer_algorithm(obj_func, lb, ub, max_iter, pop_size, num_red_deer, num_stags):
    # Initialization
    num_dim = len(lb)
    positions = np.zeros((pop_size, num_dim))
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        positions[i] = lb + (ub - lb) * np.random.rand(num_dim)
        fitness[i] = obj_func(positions[i])

    # Main loop
    for t in range(max_iter):
        # Sort the positions and fitness values
        sorted_indices = np.argsort(fitness)
        positions = positions[sorted_indices]
        fitness = fitness[sorted_indices]

        # Red deer phase
        for i in range(num_red_deer):
            # Select a random deer
            r = np.random.randint(num_stags, pop_size)

            # Move towards the centroid of the stags
            centroid = np.mean(positions[:num_stags], axis=0)
            direction = centroid - positions[r]
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            step_size = np.random.normal(0, 1) * np.exp(-t / max_iter)
            new_position = positions[r] + step_size * direction

            # Clip the position to the search space
            new_position = np.clip(new_position, lb, ub)

            # Evaluate the new position
            new_fitness = obj_func(new_position)

            # Update the position and fitness if the new position is better
            if new_fitness < fitness[r]:
                positions[r] = new_position
                fitness[r] = new_fitness

        # Stag phase
        for i in range(num_stags, pop_size):
            # Select two stags
            s1, s2 = np.random.choice(num_stags, size=2, replace=False)

            # Compute the new position
            new_position = positions[i] + np.random.uniform() * (positions[s1] - positions[s2])

            # Clip the position to the search space
            new_position = np.clip(new_position, lb, ub)

            # Evaluate the new position
            new_fitness = obj_func(new_position)

            # Update the position and fitness if the new position is better
            if new_fitness < fitness[i]:
                positions[i] = new_position
                fitness[i] = new_fitness

    # Return the best position and fitness
    best_index = np.argmin(fitness)
    best_position = positions[best_index]
    best_fitness = fitness[best_index]
    return best_position, best_fitness