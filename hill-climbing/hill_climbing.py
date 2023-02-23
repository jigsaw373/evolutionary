import random

def hill_climbing(f, n):
    # f is the fitness function
    # n is the number of dimensions of the problem
    x = [random.uniform(-10, 10) for i in range(n)] # initialize random solution
    best = x
    count = 0
    while True:
        count += 1
        neighbor = [x[i] + random.uniform(-1, 1) for i in range(n)] # generate a random neighbor
        if f(neighbor) > f(best): # if the neighbor is better, update the best solution
            best = neighbor
            x = neighbor
        if count > 1000: # stop after 1000 iterations without improvement
            break
    return best

# Example fitness function
def sphere(x):
    return sum(i**2 for i in x)

print(hill_climbing(sphere, 3))

