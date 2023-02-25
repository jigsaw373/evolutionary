import numpy as np

# Define the objective function to be optimized
def sphere(x):
    return np.sum(x**2)

# Set up the MFO algorithm
num_moths = 50
dim = 10
lb = -5.12
ub = 5.12
pa = 0.25
max_iter = 100

positions = np.random.uniform(lb, ub, size=(num_moths, dim))
fitness = np.array([sphere(pos) for pos in positions])

# Run the MFO algorithm to optimize the objective function
best_fitness = np.inf
best_position = None

for i in range(max_iter):
    # Update the brightness of each moth
    brightness = 1 / (1 + fitness)
    
    # Calculate the distance to the flame
    distances = np.sqrt(np.sum((positions - positions.mean(axis=0))**2, axis=1))
    
    # Update the position of each moth
    for j in range(num_moths):
        # Calculate the attraction to the flame
        flame_distance = distances[j]
        flame_attraction = pa * np.exp(-flame_distance) * (positions.mean(axis=0) - positions[j])
        
        # Calculate the attraction to other moths
        moth_indices = np.arange(num_moths)
        moth_indices = np.delete(moth_indices, j)
        moth_distances = distances[moth_indices]
        brightest_moth_index = np.argmax(brightness[moth_indices])
        brightest_moth_distance = moth_distances[brightest_moth_index]
        brightest_moth_position = positions[moth_indices[brightest_moth_index]]
        moth_attraction = pa * np.exp(-brightest_moth_distance) * (brightest_moth_position - positions[j])
        
        # Update the position of the moth
        movement = flame_attraction + moth_attraction
        positions[j] += movement
        
        # Clip the position to the search space
        positions[j] = np.clip(positions[j], lb, ub)
        
        # Update the fitness of the moth
        fitness[j] = sphere(positions[j])
        
        # Update the best solution found so far
        if fitness[j] < best_fitness:
            best_fitness = fitness[j]
            best_position = positions[j]
            
    # Print the current best solution
    print("Iteration", i+1, "Best cost:", best_fitness)

# Print the final best solution
print("Final best cost:", best_fitness)
print("Final best position:", best_position)