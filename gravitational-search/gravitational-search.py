import numpy as np

def gsa(f, lb, ub, dim, n_iter, n_pop, G0=100, alpha=20, G_dec=0.99):
    # f: objective function
    # lb: lower bounds
    # ub: upper bounds
    # dim: dimensions
    # n_iter: number of iterations
    # n_pop: population size
    # G0: initial gravitational constant
    # alpha: power index
    # G_dec: gravitational constant decay rate
    
    # initialization
    X = np.random.uniform(lb, ub, size=(n_pop, dim))
    G = G0
    best_fitness = np.inf
    best_solution = None
    fitness_history = []
    
    for i in range(n_iter):
        # calculate fitness
        fitness = np.array([f(x) for x in X])
        # update best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = X[best_idx]
        # update fitness history
        fitness_history.append(best_fitness)
        # calculate mass
        M = fitness / np.sum(fitness)
        # calculate acceleration
        a = np.zeros_like(X)
        for j in range(n_pop):
            for k in range(n_pop):
                if j == k:
                    continue
                r = np.linalg.norm(X[j] - X[k])
                a[j] += (X[k] - X[j]) * M[k] / r ** alpha
        # calculate new position
        X = X + np.random.uniform(size=X.shape) * a / (G + 1e-10)
        # apply boundary conditions
        X = np.clip(X, lb, ub)
        # decay gravitational constant
        G *= G_dec
    
    return best_solution, best_fitness, fitness_history