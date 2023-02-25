import numpy as np

def levy_flight(beta):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, sigma)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

def generate_cuckoos(n, dim):
    cuckoos = np.random.uniform(low=-10, high=10, size=(n, dim))
    return cuckoos

def get_best_nests(nests, fitness):
    sorted_fitness = np.argsort(fitness)
    best_nests = nests[sorted_fitness][:int(len(nests) * 0.25)]
    return best_nests

def empty_nests(nests, alpha):
    n, dim = nests.shape
    beta = 3 / 2
    sigma = (np.power(np.math.gamma(1 + beta), 2) * np.power(np.sin(np.pi * beta / 2), 2) / (np.power(np.math.gamma((1 + beta) / 2), 2) * beta * np.power(2, (beta - 1)))) ** (1 / (2 * beta))
    u = np.random.normal(0, sigma, size=(n, dim))
    v = np.random.normal(0, 1, size=(n, dim))
    step = u / np.power(np.abs(v), 1 / beta)
    step_size = alpha * step * (nests - nests.mean(axis=0))
    new_nests = nests + step_size
    return new_nests

def cuckoo_optimization(n, dim, num_iter):
    nests = generate_cuckoos(n, dim)
    fitness = np.random.uniform(low=0, high=1, size=n)
    best_nests = get_best_nests(nests, fitness)
    best_fitness = fitness.min()
    for i in range(num_iter):
        new_nests = empty_nests(nests, alpha=0.01)
        new_fitness = np.random.uniform(low=0, high=1, size=n)
        new_best_nests = get_best_nests(new_nests, new_fitness)
        for j in range(len(new_best_nests)):
            if new_fitness[j] < best_fitness:
                idx = np.random.randint(len(best_nests))
                best_nests[idx] = new_best_nests[j]
                best_fitness = new_fitness[j]
        nests = new_nests
        fitness = new_fitness
    return best_nests, best_fitness