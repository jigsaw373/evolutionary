import numpy as np

def hho(objective_func, lb, ub, dim, search_agents=50, max_iter=100):
    # Initialize the population of hawks
    x = np.random.uniform(lb, ub, (search_agents, dim))
    
    # Initialize the fitness of each hawk
    fitness = np.zeros(search_agents)
    for i in range(search_agents):
        fitness[i] = objective_func(x[i, :])
        
    # Find the index of the imperialist hawk (the one with the best fitness)
    imp = np.argmin(fitness)
    
    # Initialize the position of the imperialist hawk
    imp_pos = np.copy(x[imp, :])
    
    # Initialize the position of the colonies
    col_pos = np.copy(x)
    
    for t in range(max_iter):
        # Calculate the fitness of each colony
        for i in range(search_agents):
            # Determine the current distance between the colony and the imperialist hawk
            dist = np.sqrt(np.sum((col_pos[i, :] - imp_pos) ** 2))
            
            # Choose a strategy based on the distance
            if dist > 0.1:
                # Explore the search space
                r1 = np.random.uniform(size=dim)
                r2 = np.random.uniform(size=dim)
                col_pos[i, :] = (col_pos[i, :] + r1 * (imp_pos - dist * r2)) / 2
            else:
                # Exploit the current best solution
                r3 = np.random.uniform(size=dim)
                col_pos[i, :] = imp_pos - r3 * (2 * lb + ub) / 3
        
        # Update the fitness of each colony
        for i in range(search_agents):
            fitness[i] = objective_func(col_pos[i, :])
        
        # Find the index of the best colony
        best_col = np.argmin(fitness)
        
        # Check if the best colony is better than the imperialist hawk
        if fitness[best_col] < fitness[imp]:
            imp_pos = np.copy(col_pos[best_col, :])
            imp = np.copy(best_col)
        
        # Update the position of the colonies based on the position of the imperialist hawk
        for i in range(search_agents):
            if i != imp:
                # Calculate the distance between the current colony and the imperialist hawk
                dist = np.sqrt(np.sum((col_pos[i, :] - imp_pos) ** 2))
                
                # Choose a strategy based on the distance
                if dist > 0.1:
                    # Attack the imperialist hawk
                    r1 = np.random.uniform(size=dim)
                    r2 = np.random.uniform(size=dim)
                    col_pos[i, :] = col_pos[i, :] + r1 * (imp_pos - dist * r2)
                else:
                    # Invade the territory of the imperialist hawk
                    r3 = np.random.uniform(size=dim)
                    col_pos[i, :] = imp_pos + r3 * (2 * lb + ub) / 3
        
        # Update the fitness of each colony
        for i in range(search_agents):
            fitness[i] = objective_func(col_pos[i, :])
    
    # Return the position of the best solution found
    return imp_pos