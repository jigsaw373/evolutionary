import random
import numpy as np

class HarmonySearch:
    def __init__(self, obj_function, lb, ub, num_harmony=10, max_iter=1000, bw=0.01, hmcr=0.9, par=0.4):
        self.obj_function = obj_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_harmony = num_harmony
        self.max_iter = max_iter
        self.bw = bw
        self.hmcr = hmcr
        self.par = par
        
    def optimize(self):
        dim = len(self.lb)
        # Initialize harmonies
        harmonies = np.zeros((self.num_harmony, dim))
        for i in range(self.num_harmony):
            harmonies[i] = self.lb + (self.ub - self.lb) * np.random.rand(dim)
        
        # Evaluate harmonies
        obj_vals = np.array([self.obj_function(h) for h in harmonies])
        
        # Sort harmonies by objective value
        sorted_idx = np.argsort(obj_vals)
        harmonies = harmonies[sorted_idx]
        obj_vals = obj_vals[sorted_idx]
        
        # Search for new harmonies
        for t in range(1, self.max_iter+1):
            new_harmonies = np.zeros((self.num_harmony, dim))
            for i in range(self.num_harmony):
                if np.random.rand() < self.hmcr:  # Memory consideration
                    idx = np.random.choice(self.num_harmony, 1)[0]
                    new_harmonies[i] = harmonies[idx]
                    if np.random.rand() < self.par:  # Pitch adjustment
                        for j in range(dim):
                            if np.random.rand() < 0.5:
                                new_harmonies[i][j] += np.random.rand() * self.bw
                            else:
                                new_harmonies[i][j] -= np.random.rand() * self.bw
                else:  # Random selection
                    new_harmonies[i] = self.lb + (self.ub - self.lb) * np.random.rand(dim)
            
            # Evaluate new harmonies
            new_obj_vals = np.array([self.obj_function(h) for h in new_harmonies])
            
            # Sort new harmonies by objective value
            all_harmonies = np.concatenate((harmonies, new_harmonies), axis=0)
            all_obj_vals = np.concatenate((obj_vals, new_obj_vals), axis=0)
            sorted_idx = np.argsort(all_obj_vals)
            all_harmonies = all_harmonies[sorted_idx]
            all_obj_vals = all_obj_vals[sorted_idx]
            
            # Keep top num_harmony harmonies
            harmonies = all_harmonies[:self.num_harmony]
            obj_vals = all_obj_vals[:self.num_harmony]
        
        # Return best harmony and its objective value
        return harmonies[0], obj_vals[0]