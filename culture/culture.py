import random
import math

# Define the function to be optimized
def func(x):
    return -x * math.sin(math.sqrt(abs(x)))

# Define the parameters of the algorithm
pop_size = 50
num_subcultures = 5
max_iterations = 100
p_mig = 0.1
p_mut = 0.1

# Initialize the population and subcultures
pop = [[random.uniform(-10, 10)] for i in range(pop_size)]
subcultures = [[] for i in range(num_subcultures)]
for i in range(pop_size):
    subcultures[i % num_subcultures].append(pop[i])

# Main loop of the algorithm
for iteration in range(max_iterations):
    # Evaluate the fitness of each individual in each subculture
    fitness = [[func(individual[0])] for individual in pop]
    sub_fitness = [[] for i in range(num_subcultures)]
    for i in range(pop_size):
        sub_fitness[i % num_subcultures].append(fitness[i])

    # Allow individuals in a subculture to interact and share their knowledge
    for i in range(num_subcultures):
        for j in range(len(subcultures[i])):
            for k in range(j+1, len(subcultures[i])):
                if sub_fitness[i][j][0] < sub_fitness[i][k][0]:
                    subcultures[i][j] = subcultures[i][k]

    # Allow individuals to migrate between subcultures
    for i in range(pop_size):
        if random.random() < p_mig:
            source = i % num_subcultures
            target = (i + random.randint(1, num_subcultures-1)) % num_subcultures
            if sub_fitness[target] and fitness[i][0] > max(sub_fitness[target])[0]:
                subcultures[source].remove(pop[i])
                subcultures[target].append(pop[i])

    # Generate new individuals through crossover and mutation
    new_pop = []
    for i in range(pop_size):
        subculture = subcultures[i % num_subcultures]
        parent1 = random.choice(subculture)
        parent2 = random.choice(subculture)
        child = [parent1[0] + random.uniform(-1, 1) * (parent1[0] - parent2[0])]
        if random.random() < p_mut:
            child[0] += random.gauss(0, 1)
        new_pop.append(child)

    # Evaluate the fitness of the new individuals
    new_fitness = [func(individual[0]) for individual in new_pop]

    # Select the best individuals in each subculture to form the new generation
    new_subcultures = [[] for i in range(num_subcultures)]
    for i in range(pop_size):
        new_subcultures[i % num_subcultures].append([new_pop[i]])

    for i in range(num_subcultures):
        new_subcultures[i].sort(key=lambda x: func(x[0]), reverse=True)
        pop[i::num_subcultures] = new_subcultures[i][:pop_size//num_subcultures]

# Print the best solution found
best_solution = max(pop, key=lambda x: func(x[0]))
print("Best solution:", best_solution[0])