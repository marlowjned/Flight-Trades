# config_loader.py
# Loads all data from config file with custom YAML expressions

import yaml
import numpy as np


def _linspace_constructor(loader, node):
    values = loader.construct_sequence(node)
    start, stop, n = values
    return np.linspace(start, stop, int(n)).tolist()


class _SimLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor("!linspace", _linspace_constructor)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=_SimLoader)
