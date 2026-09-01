import numpy as np

def evaluate(*args):
    min_mass, max_mass = args
    return np.linspace(min_mass, max_mass, 10).tolist()
