# config_loader.py
# Loads all data from config file with custom YAML expressions

import yaml
import numpy as np
import importlib.util
import os

CUSTOM_EXPR_DIR = "user_inputs/custom-expressions"


def _linspace_constructor(loader, node):
    values = loader.construct_sequence(node)
    start, stop, n = values
    return np.linspace(start, stop, int(n)).tolist()


def _expr_constructor(loader, node):
    values = loader.construct_sequence(node)
    script_rel_path = values[0]
    args = values[1:]

    full_path = os.path.join(CUSTOM_EXPR_DIR, script_rel_path)
    spec = importlib.util.spec_from_file_location("_custom_expr", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate(*args)


class _SimLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor("!linspace", _linspace_constructor)
        self.add_constructor("!expr", _expr_constructor)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=_SimLoader)
