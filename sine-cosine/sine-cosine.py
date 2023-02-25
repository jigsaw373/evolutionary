import numpy as np

def sine_cosine_algorithm(fitness_func, lb, ub, pop_size=50, max_iter=100):
    # initialize the population
    pop = np.random.uniform(lb, ub, (pop_size, len(lb)))
    best = pop[np.argmin([fitness_func(x) for x in pop])]

    for i in range(max_iter):
        # generate random numbers between -1 and 1 using the sine function
        rand_sin = np.sin(np.random.uniform(-np.pi/2, np.pi/2, (pop_size, len(lb))))
        # generate random numbers between 0 and 1 using the cosine function
        rand_cos = np.cos(np.random.uniform(0, 2*np.pi, (pop_size, len(lb))))
        # update the positions of the population
        pop = pop + rand_sin * (best - pop) + rand_cos * (pop - pop.mean(axis=0))
        # apply bounds to the positions
        pop = np.clip(pop, lb, ub)
        # evaluate the fitness of the population
        fitness = [fitness_func(x) for x in pop]
        # update the global best solution
        if min(fitness) < fitness_func(best):
            best = pop[np.argmin(fitness)]
    return best