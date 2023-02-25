import numpy as np

def SHO(obj_func, lb, ub, num_hyenas=50, max_iter=100):
   
    # Initialization
    dim = len(lb)
    hyenas = np.random.uniform(lb, ub, (num_hyenas, dim))
    fitness = np.zeros(num_hyenas)

    # Main loop
    for t in range(max_iter):
        # Evaluate fitness
        for i in range(num_hyenas):
            fitness[i] = obj_func(hyenas[i, :])

        # Find the best hyena
        best_hyena_idx = np.argmin(fitness)
        best_hyena = hyenas[best_hyena_idx, :]

        # Update hyenas
        for i in range(num_hyenas):
            if i == best_hyena_idx:
                continue

            # Calculate distance
            dist = np.linalg.norm(hyenas[i, :] - best_hyena)

            # Update position
            if fitness[i] < fitness[best_hyena_idx]:
                a = 2.0 - 2.0 * t / max_iter
                b = np.random.uniform(0, 1)
                hyenas[i, :] += a * np.exp(-b * dist) * (hyenas[i, :] - best_hyena)
            else:
                a = 2.0 * t / max_iter
                b = np.random.uniform(0, 1)
                hyenas[i, :] += a * np.exp(-b * dist) * (hyenas[i, :] - best_hyena)

            # Enforce bounds
            hyenas[i, :] = np.maximum(hyenas[i, :], lb)
            hyenas[i, :] = np.minimum(hyenas[i, :], ub)

    # Find the best solution
    best_hyena_idx = np.argmin(fitness)
    best_hyena = hyenas[best_hyena_idx, :]
    best_fitness = fitness[best_hyena_idx]

    return (best_hyena, best_fitness)