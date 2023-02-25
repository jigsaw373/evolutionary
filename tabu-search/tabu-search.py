import random

def tabu_search(f, n, tabulen, maxiter):
    # f is the fitness function
    # n is the number of dimensions of the problem
    # tabulen is the length of the tabu list
    # maxiter is the maximum number of iterations
    x = [random.uniform(-10, 10) for i in range(n)] # initialize random solution
    best = x
    tabu = [x] * tabulen # initialize tabu list
    count = 0
    while count < maxiter:
        count += 1
        candidate_list = []
        for i in range(n):
            for j in [-1, 1]:
                candidate = x.copy()
                candidate[i] += j
                if candidate not in tabu:
                    candidate_list.append(candidate)
        if len(candidate_list) == 0:
            break
        candidate_list.sort(key=f, reverse=True)
        x = candidate_list[0]
        if f(x) > f(best):
            best = x
        tabu.pop(0)
        tabu.append(x)
    return best

# Example fitness function
def sphere(x):
    return sum(i**2 for i in x)

print(tabu_search(sphere, 3, 10, 100))