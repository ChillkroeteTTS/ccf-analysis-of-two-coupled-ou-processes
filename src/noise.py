from enum import Enum

import numpy as np

class NoiseType(Enum):
    WHITE = 1
    RED = 2

def white_noise(n):
    return np.random.rand(n)*2 - 1
