# ConfigLoader.py
# Loads all data from config file with custom YAML expressions

import yaml
import numpy as np


# Custom tag handlers

def linspace_constructor(loader, node):
	values = loader.construct_sequence(node)
	start, stop, n = values
	return np.linspace(start, stop, int(n)).tolist()


# Register tags on a custom loader

class SimLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor("!linspace", linspace_constructor)

def loadConfig(path: str) -> dict:
	with open(path, "r") as f:
		return yaml.load(f, Loader=SimLoader)



