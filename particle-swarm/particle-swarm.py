import numpy as np
import pyswarms as ps

# Define the objective function to be optimized
def sphere(x):
    return np.sum(x**2)

# Define the bounds of the search space
bounds = (np.array([-5, -5]), np.array([5, 5]))

# Set up the PSO optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, bounds=bounds)

# Run the PSO algorithm to optimize the objective function
best_cost, best_pos = optimizer.optimize(sphere, iters=100)

# Print the results
print("Best cost: ", best_cost)
print("Best position: ", best_pos)