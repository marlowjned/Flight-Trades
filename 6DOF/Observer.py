# Observer.py
# Handles quick implementation of data collection, essential for a dynamic flight trade framework

# Default includes: all state variables, mass, Ixx Iyy, CP, CG, environmental data, g, stability

class Observer:
    def __init__(self):
        self._listeners = {}  # Variable name -> list of callbacks

    def subscribe(self, variable_name, callback):
        if variable_name not in self._listeners:
            self._listeners[variable_name] = []
        self._listeners[variable_name].append(callback)

    def notify(self, variable_name, value, t):
        for cb in self._listeners.get(variable_name, []):
            cb(t, value)

# TODO: data exporting



