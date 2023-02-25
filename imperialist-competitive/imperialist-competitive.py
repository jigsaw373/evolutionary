import numpy as np

def imperialist_competitive_algorithm(objective_func, bounds, n_countries=10, n_iterations=1000, revolution_rate=0.1, assimilation_rate=0.1):
    # Initialize the population of countries with random solutions
    n_variables = len(bounds)
    countries = np.zeros((n_countries, n_variables))
    for i in range(n_countries):
        for j in range(n_variables):
            countries[i,j] = np.random.uniform(bounds[j][0], bounds[j][1])
    
    for iteration in range(n_iterations):
        # Evaluate the fitness of each country
        fitness = np.array([objective_func(country) for country in countries])
        imperialist_index = np.argmax(fitness)
        
        # Find the imperialist
        imperialist = countries[imperialist_index]
        imperialist_fitness = fitness[imperialist_index]
        
        # For each colony, calculate its normalized distance from the imperialist
        distances = np.zeros(n_countries)
        for i in range(n_countries):
            distances[i] = np.linalg.norm(countries[i] - imperialist) / np.linalg.norm(bounds)
        
        # Calculate the probability of revolution for each colony
        p = revolution_rate * (1 - distances)
        p[imperialist_index] = 0
        p = p / np.sum(p)
        
        # If a colony initiates a revolution, replace the imperialist with the colony and reset the hierarchy
        for i in range(n_countries):
            if np.random.rand() < p[i]:
                countries[imperialist_index] = countries[i]
                fitness[imperialist_index] = fitness[i]
                break
        
        # If no revolution occurs, redistribute some of the imperialist's resources to the weaker colonies
        if np.random.rand() > assimilation_rate:
            for i in range(n_countries):
                if i != imperialist_index:
                    delta = np.random.uniform(0, 1) * (imperialist - countries[i])
                    countries[i] += delta
        
        # Evaluate the fitness of each colony and update the population
        for i in range(n_countries):
            if i != imperialist_index:
                fitness[i] = objective_func(countries[i])
        
        # Print the best solution found so far
        best_fitness = np.max(fitness)
        best_solution = countries[np.argmax(fitness)]
        print("Iteration {}: Best fitness = {}".format(iteration, best_fitness))
        
    return best_solution, best_fitness