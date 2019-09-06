import functools
from typing import List

import numpy as np


def ou_step(init_cond, tau, history: List, n):
    hits_size = len(history)
    if hits_size > 0:
        prev_x = history[hits_size - 1]
        x = -prev_x/tau + np.sqrt(2 / tau) * n
    else:
        x = -init_cond/tau + np.sqrt(2 / tau) * n
    history.append(x)
    return history

def ou(noise_arr, tau, initial_condition=0):
    return np.array(functools.reduce(functools.partial(ou_step, initial_condition, tau),
                            noise_arr,
                            []))
