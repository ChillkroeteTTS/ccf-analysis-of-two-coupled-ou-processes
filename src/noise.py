import functools
from enum import Enum

import numpy as np

class NoiseType(Enum):
    WHITE = 1
    RED = 2

def white_noise(n):
    return np.random.normal(0, 1, n)

def red_noise(r, n):
    noise = white_noise(n)

    for i in range(1, len(noise)):
        noise[i] = r * noise[i-1] + np.sqrt(1 - np.power(r, 2)) * noise[i]

    return noise
