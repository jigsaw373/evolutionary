import numpy as np
import math

def whale_optimization_algorithm(cost_func, dim, n, iterations, lb, ub):
    # Initialize positions and velocities of search agents
    positions = np.zeros((n, dim))
    for i in range(n):
        positions[i, :] = np.random.uniform(0, 1, dim) * (ub - lb) + lb
    
    # Initialize convergence curve
    convergence_curve = np.zeros(iterations)
    
    # Main loop
    for t in range(iterations):
        # Update a
        a = 2 - t * ((2) / iterations)
        
        for i in range(n):
            # Update the position of the current whale
            r1 = np.random.uniform(0, 1, dim)
            r2 = np.random.uniform(0, 1, dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a - 1) * np.random.uniform(0, 1) + 1
            
            p = np.random.uniform(0, 1, dim)
            d = np.abs(C * positions[np.random.randint(0, n), :] - positions[i, :])
            new_position = np.clip(p * positions[i, :] + b * d * np.exp(l * A), lb, ub)
            
            # Evaluate the new position
            new_cost = cost_func(new_position)
            
            # Update the position if it is better
            if new_cost < cost_func(positions[i, :]):
                positions[i, :] = new_position
        
        # Update convergence curve
        convergence_curve[t] = cost_func(positions.min(axis=0))
    
    # Return the best solution and convergence curve
    best_solution = positions.min(axis=0)
    return best_solution, convergence_curve