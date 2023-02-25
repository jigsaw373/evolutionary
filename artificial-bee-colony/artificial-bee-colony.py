import numpy as np

def abc_algorithm(objective_function, lb, ub, colony_size=10, max_iter=100):
    dim = len(lb)
    best_solution = None
    best_fitness = np.inf
    
    # initialize population
    population = np.zeros((colony_size, dim))
    for i in range(colony_size):
        population[i] = np.random.uniform(lb, ub)
    
    # evaluate population fitness
    fitness = np.zeros(colony_size)
    for i in range(colony_size):
        fitness[i] = objective_function(population[i])
    
    # main loop
    for it in range(max_iter):
        # employed bee phase
        for i in range(colony_size):
            phi = np.random.uniform(low=-1, high=1, size=dim)
            new_solution = population[i] + phi * (population[np.random.randint(colony_size)] - population[i])
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = objective_function(new_solution)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
        
        # onlooker bee phase
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        for i in range(colony_size):
            if np.random.uniform() < probabilities[i]:
                phi = np.random.uniform(low=-1, high=1, size=dim)
                new_solution = population[i] + phi * (population[np.random.randint(colony_size)] - population[i])
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = objective_function(new_solution)
                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
        
        # scout bee phase
        for i in range(colony_size):
            if np.random.uniform() < 0.1:
                population[i] = np.random.uniform(lb, ub)
                fitness[i] = objective_function(population[i])
        
        # update best solution
        index = np.argmin(fitness)
        if fitness[index] < best_fitness:
            best_fitness = fitness[index]
            best_solution = population[index]
            
        print("Iteration {}: Best Fitness = {}".format(it, best_fitness))
        
    return best_solution, best_fitness