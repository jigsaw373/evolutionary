import numpy as np

def init_bats(num_bats, num_dimensions, lb, ub):
    """Initialize the bats with random positions and velocities"""
    bats = np.zeros((num_bats, num_dimensions + 2)) # create an array to hold the bats
    bats[:, :num_dimensions] = np.random.uniform(lb, ub, (num_bats, num_dimensions)) # initialize positions
    bats[:, num_dimensions] = np.random.uniform(0, 1, num_bats) # initialize velocities
    bats[:, num_dimensions+1] = np.inf # set initial fitness to infinity
    return bats

def simple_bounds(pos, lb, ub):
    """Apply simple bounds to the bat's position"""
    pos[pos < lb] = lb[pos < lb]
    pos[pos > ub] = ub[pos > ub]
    return pos

def update_velocity(bat, global_best, A, r, alpha, gamma):
    """Update the bat's velocity"""
    bat_velocity = bat[-2]
    bat_position = bat[:-2]
    rand_vect = np.random.uniform(0, 1, len(bat_position))
    velocity = bat_velocity + (bat_position - global_best) * A
    velocity += alpha * (rand_vect - 0.5)
    velocity *= gamma
    return velocity

def update_position(position, velocity):
    """Update the bat's position"""
    position += velocity
    return position

def update_pulse_rate(bat, f_min, f_max, r):
    """Update the pulse rate of the bat"""
    pulse_rate = f_min + (f_max - f_min) * r
    return pulse_rate

def update_loudness(loudness, alpha):
    """Update the loudness of the bat"""
    loudness *= alpha
    return loudness

def update_fitness(bat, func):
    """Update the fitness of the bat"""
    fitness = func(bat[:-2])
    if fitness < bat[-1]:
        bat[-1] = fitness
    return bat

def bat_algorithm(func, num_bats, num_iterations, num_dimensions, lb, ub, A, f_min, f_max, alpha, gamma):
    """Bat algorithm for minimizing a function"""
    bats = init_bats(num_bats, num_dimensions, lb, ub)
    global_best = np.zeros(num_dimensions)
    global_best_fitness = np.inf
    for i in range(num_iterations):
        for j in range(num_bats):
            bat = bats[j]
            frequency = f_min + (f_max - f_min) * np.random.uniform(0, 1)
            velocity = update_velocity(bat, global_best, A, frequency, alpha, gamma)
            position = update_position(bat[:-2], velocity)
            position = simple_bounds(position, lb, ub)
            bat[:-2] = position
            bat[-2] = frequency
            bat = update_fitness(bat, func)
            if bat[-1] < global_best_fitness:
                global_best = bat[:-2]
                global_best_fitness = bat[-1]
            bats[j] = bat
        A *= gamma
        alpha *= 0.9 # damping factor for loudness
    return global_best, global_best_fitness