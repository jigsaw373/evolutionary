import numpy as np

# Define the objective function
def objective(x):
    return np.sum(x**2)

# Define the Grey Wolf Optimizer function
def grey_wolf_optimizer(obj_func, lb, ub, dim, search_agents=10, max_iter=100):
    # Initialize alpha, beta, and delta positions
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")
    
    # Initialize the search agents
    positions = np.zeros((search_agents, dim))
    for i in range(search_agents):
        positions[i] = np.random.uniform(lb, ub, dim)
    
    # Main loop
    for iteration in range(max_iter):
        # Update alpha, beta, and delta positions
        for i in range(search_agents):
            # Evaluate the objective function for the current position
            fitness = obj_func(positions[i])
            
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i]
            elif (fitness > alpha_score and fitness < beta_score):
                beta_score = fitness
                beta_pos = positions[i]
            elif (fitness > alpha_score and fitness > beta_score and fitness < delta_score):
                delta_score = fitness
                delta_pos = positions[i]
        
        # Update the search agents
        a = 2 - iteration * (2 / max_iter)
        for i in range(search_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha
            
            r1 = np.random.random()
            r2 = np.random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta
            
            r1 = np.random.random()
            r2 = np.random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta
            
            positions[i] = (X1 + X2 + X3) / 3
            
    return alpha_pos, alpha_score