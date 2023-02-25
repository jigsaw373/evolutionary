import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.backend.topology import Pyramid
from pyswarms.backend.operators import compute_pbest

# Define the fitness function to be optimized
def sphere(x):
    return fx.sphere(x)

# Set the search space and the number of dimensions
bounds = ([-5, -5, -5], [5, 5, 5])
dimensions = 3

# Initialize the swarm and the topology
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
swarm = ps.discrete.binary.BinaryPSO(n_particles=20, dimensions=dimensions, options=options, bounds=bounds)
topology = Pyramid(static=False)

# Initialize the dragonfly algorithm
dragonfly = ps.single.DragonflySwarmOptimizer(n_particles=20, dimensions=dimensions, options=options, bounds=bounds, topology=topology)

# Optimize the function using both algorithms and compare the results
sphere_min = fx.sphere([0]*dimensions)

for i in range(100):
    # Run one iteration of the PSO algorithm
    cost, pos = swarm.optimize(sphere, iters=1)

    # Compute the personal best for each particle
    swarm.pbest_pos, swarm.pbest_cost = compute_pbest(swarm)

    # Run one iteration of the dragonfly algorithm
    dragonfly.optimize(sphere)

    # Compare the results
    print('Iteration:', i, 'PSO Best:', swarm.best_cost, 'DA Best:', dragonfly.best_cost)
    if swarm.best_cost <= sphere_min and dragonfly.best_cost <= sphere_min:
        break