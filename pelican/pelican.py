import numpy as np

class PelicanOptimizer:
    def __init__(self, obj_func, lb, ub, num_pelicans=30, max_iter=500, alpha=0.5, beta=1.5, gamma=0.1):
        self.obj_func = obj_func
        self.lb = lb
        self.ub = ub
        self.num_vars = len(lb)
        self.num_pelicans = num_pelicans
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_solution = None
        self.best_fitness = np.inf
    
    def optimize(self):
        # Initialization
        positions = np.random.uniform(self.lb, self.ub, size=(self.num_pelicans, self.num_vars))
        fitness = np.array([self.obj_func(p) for p in positions])
        best_pelican_index = np.argmin(fitness)
        self.best_solution = positions[best_pelican_index]
        self.best_fitness = fitness[best_pelican_index]
        
        # Main loop
        for t in range(self.max_iter):
            # Update the leader pelican
            if t % 10 == 0:
                best_pelican_index = np.argmin(fitness)
                if fitness[best_pelican_index] < self.best_fitness:
                    self.best_solution = positions[best_pelican_index]
                    self.best_fitness = fitness[best_pelican_index]
            
            # Move the pelicans
            for i in range(self.num_pelicans):
                # Determine the leaders
                r1 = np.random.randint(self.num_pelicans)
                r2 = np.random.randint(self.num_pelicans)
                leader1 = positions[r1]
                leader2 = positions[r2]
                if fitness[r1] < fitness[r2]:
                    leader = leader1
                else:
                    leader = leader2
                
                # Update the position of the pelican
                r = np.random.uniform()
                if r < self.alpha:
                    # Move towards the leader
                    direction = leader - positions[i]
                    positions[i] += np.random.uniform() * direction
                elif r < self.beta:
                    # Move randomly
                    positions[i] += np.random.uniform(-1, 1, size=self.num_vars) * (self.ub - self.lb) * self.gamma
                else:
                    # Move towards the best pelican
                    direction = self.best_solution - positions[i]
                    positions[i] += np.random.uniform() * direction
                
                # Apply bounds
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Update the fitness
                fitness[i] = self.obj_func(positions[i])
        
        return self.best_solution, self.best_fitness