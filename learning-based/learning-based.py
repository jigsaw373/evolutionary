import numpy as np

# Define the fitness function
def fitness(x):
    return np.sum(np.square(x))

# Define the TLBO algorithm
def tlbo(fitness, bounds, n, m, max_iter):
    # Initialize the population
    pop = np.random.uniform(bounds[0], bounds[1], size=(n, m))
    # Evaluate the fitness of each individual
    fit = np.apply_along_axis(fitness, 1, pop)
    # Main loop
    for t in range(max_iter):
        # Determine the best individuals (teachers)
        idx = np.argsort(fit)[:n//2]
        teachers = pop[idx, :]
        # Teaching and learning process
        for i in range(n):
            # Select a teacher randomly
            j = np.random.choice(n//2)
            # Select a student randomly
            k = np.random.choice(n)
            while k == i:
                k = np.random.choice(n)
            # Teaching and learning process
            r = np.random.uniform(-1, 1, size=m)
            x_new = pop[i, :] + r * (teachers[j, :] - pop[i, :]) + r * (pop[k, :] - pop[i, :])
            # Evaluate the fitness of the new individual
            fit_new = fitness(x_new)
            # Update the student if the fitness improves
            if fit_new < fit[i]:
                pop[i, :] = x_new
                fit[i] = fit_new
    # Return the best individual and its fitness
    idx = np.argmin(fit)
    return pop[idx, :], fit[idx]