import math
import random

def simulated_annealing(f, n, T, cool_rate):
    # f is the fitness function
    # n is the number of dimensions of the problem
    # T is the initial temperature
    # cool_rate is the cooling rate
    x = [random.uniform(-10, 10) for i in range(n)] # initialize random solution
    best = x
    count = 0
    while T > 1e-8: # stop when temperature is close to zero
        count += 1
        neighbor = [x[i] + random.uniform(-1, 1) for i in range(n)] # generate a random neighbor
        delta = f(neighbor) - f(x)
        if delta > 0: # if the neighbor is better, update the best solution
            x = neighbor
            if f(neighbor) > f(best):
                best = neighbor
        else:
            prob = math.exp(delta / T)
            if random.random() < prob: # accept the neighbor with a probability
                x = neighbor
        T *= cool_rate # reduce the temperature
    return best

# Example fitness function
def sphere(x):
    return sum(i**2 for i in x)

print(simulated_annealing(sphere, 3, 10, 0.99))