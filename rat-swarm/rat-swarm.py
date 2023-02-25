import random
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class RSO:
    def __init__(self, num_rats, num_iterations, search_space, objective_function):
        self.num_rats = num_rats
        self.num_iterations = num_iterations
        self.search_space = search_space
        self.objective_function = objective_function
        self.best_solution = None
        self.best_fitness = float('inf')
        self.rats = self.initialize_rats()

    def initialize_rats(self):
        rats = []
        for i in range(self.num_rats):
            x = random.uniform(self.search_space[0], self.search_space[1])
            y = random.uniform(self.search_space[0], self.search_space[1])
            rats.append({'x': x, 'y': y})
        return rats

    def run(self):
        for i in range(self.num_iterations):
            for rat in self.rats:
                rat['fitness'] = self.objective_function(rat['x'], rat['y'])
                if rat['fitness'] < self.best_fitness:
                    self.best_fitness = rat['fitness']
                    self.best_solution = rat.copy()

            for rat in self.rats:
                other_rats = [r for r in self.rats if r != rat]
                random_rat = random.choice(other_rats)
                if rat['fitness'] < random_rat['fitness']:
                    distance_to_random_rat = distance(rat['x'], rat['y'], random_rat['x'], random_rat['y'])
                    dx = (random_rat['x'] - rat['x']) / distance_to_random_rat
                    dy = (random_rat['y'] - rat['y']) / distance_to_random_rat
                    step_size = random.uniform(0, 1) * distance_to_random_rat
                    new_x = rat['x'] + step_size * dx
                    new_y = rat['y'] + step_size * dy
                    if self.search_space[0] <= new_x <= self.search_space[1] and self.search_space[0] <= new_y <= self.search_space[1]:
                        rat['x'] = new_x
                        rat['y'] = new_y

    def print_result(self):
        print(f"Best solution: {self.best_solution}")
        print(f"Best fitness: {self.best_fitness}")

def objective_function(x, y):
    return (x - 1) ** 2 + (y - 2) ** 2

rso = RSO(num_rats=10, num_iterations=100, search_space=(-10, 10), objective_function=objective_function)
rso.run()
rso.print_result()