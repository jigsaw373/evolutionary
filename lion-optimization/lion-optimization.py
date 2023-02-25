import numpy as np

class LionOptimizationAlgorithm:
    def __init__(self, objective_function, lb, ub, dimension, population_size, p=0.5, ub_init=2.0):
        self.objective_function = objective_function
        self.lb = lb
        self.ub = ub
        self.dimension = dimension
        self.population_size = population_size
        self.p = p
        self.ub_init = ub_init

    def optimize(self, max_iter):
        # initialize the population
        positions = np.random.uniform(low=self.lb, high=self.ub, size=(self.population_size, self.dimension))
        ub_matrix = np.tile(self.ub_init * (self.ub - self.lb), (self.population_size, self.dimension))
        fitness = np.apply_along_axis(self.objective_function, 1, positions)
        best_fitness_idx = np.argmin(fitness)
        best_fitness = fitness[best_fitness_idx]
        best_position = positions[best_fitness_idx].copy()

        for i in range(max_iter):
            # sort the population by fitness
            sorted_idx = np.argsort(fitness)
            positions = positions[sorted_idx]
            fitness = fitness[sorted_idx]

            # divide the population into two groups: lions and cubs
            n_lions = int(self.p * self.population_size)
            lions = positions[:n_lions]
            cubs = positions[n_lions:]

            # update the position of the lions
            mean_lion = np.mean(lions, axis=0)
            new_lions = np.zeros_like(lions)
            for j in range(n_lions):
                r = np.random.uniform(size=self.dimension)
                new_lions[j] = lions[j] + r * (mean_lion - lions[j])
                new_lions[j] = np.clip(new_lions[j], self.lb, self.ub)
            new_lions_fitness = np.apply_along_axis(self.objective_function, 1, new_lions)
            lions_mask = new_lions_fitness < fitness[:n_lions]
            lions[lions_mask] = new_lions[lions_mask]
            fitness[:n_lions] = new_lions_fitness[lions_mask]

            # update the position of the cubs
            for j in range(n_lions, self.population_size):
                r = np.random.uniform(size=self.dimension)
                a = 2 * r - 1
                b = np.abs(a * lions[np.random.randint(n_lions)] - cubs[j])
                new_cub = cubs[j] + a * b
                new_cub = np.clip(new_cub, self.lb, self.ub)
                new_cub_fitness = self.objective_function(new_cub)
                if new_cub_fitness < fitness[j]:
                    cubs[j] = new_cub
                    fitness[j] = new_cub_fitness

            # update the global best position
            if fitness.min() < best_fitness:
                best_fitness_idx = np.argmin(fitness)
                best_fitness = fitness[best_fitness_idx]
                best_position = positions[best_fitness_idx].copy()

        return best_position, 