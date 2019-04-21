import numpy as np


def sign(x):
    return np.sign(x)


# Don't know if this works
def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()