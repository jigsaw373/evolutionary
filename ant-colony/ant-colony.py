import numpy as np

class AntColony:
    def __init__(self, dist, num_ants, num_iterations, decay, alpha=1, beta=2):
        self.dist = dist
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones_like(dist) / len(dist)
        self.best_path = None
        self.best_dist = np.inf
        
    def run(self):
        for i in range(self.num_iterations):
            paths = self.generate_paths()
            self.update_pheromone(paths)
            best_path, best_dist = self.get_best_path(paths)
            if best_dist < self.best_dist:
                self.best_path = best_path
                self.best_dist = best_dist
            self.pheromone = self.pheromone * self.decay
        
    def generate_paths(self):
        paths = []
        for ant in range(self.num_ants):
            path = []
            visited = set()
            node = np.random.choice(len(self.dist))
            path.append(node)
            visited.add(node)
            while len(visited) < len(self.dist):
                probs = self.get_probabilities(node, visited)
                node = np.random.choice(len(self.dist), p=probs)
                path.append(node)
                visited.add(node)
            paths.append(path)
        return paths
    
    def get_probabilities(self, node, visited):
        pheromone = np.copy(self.pheromone[node])
        pheromone[list(visited)] = 0
        if np.sum(pheromone) == 0:
            return np.ones_like(pheromone) / len(pheromone)
        else:
            dist = self.dist[node]
            return (pheromone ** self.alpha) * ((1 / dist) ** self.beta) / np.sum((pheromone ** self.alpha) * ((1 / dist) ** self.beta))
        
    def update_pheromone(self, paths):
        pheromone_delta = np.zeros_like(self.pheromone)
        for path in paths:
            for i in range(len(path) - 1):
                pheromone_delta[path[i], path[i+1]] += 1 / self.dist[path[i], path[i+1]]
        self.pheromone = self.pheromone + pheromone_delta
    
    def get_best_path(self, paths):
        best_path = None
        best_dist = np.inf
        for path in paths:
            dist = 0
            for i in range(len(path) - 1):
                dist += self.dist[path[i], path[i+1]]
            if dist < best_dist:
                best_path = path
                best_dist = dist
        return best_path, best_dist