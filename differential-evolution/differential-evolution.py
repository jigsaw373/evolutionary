import numpy as np
from scipy.optimize import differential_evolution

# Define the objective function to be optimized
def sphere(x):
    return np.sum(x**2)

# Define the bounds of the search space
bounds = [(-5, 5), (-5, 5)]

# Set up the DE optimizer
optimizer = differential_evolution(sphere, bounds)

# Run the DE algorithm to optimize the objective function
result = optimizer.run()

# Print the results
print("Best cost: ", result.fun)
print("Best position: ", result.x)