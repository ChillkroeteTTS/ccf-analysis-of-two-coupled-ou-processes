import functools
from typing import List

import numpy as np


def ou_step(init_cond, tau, history: List, n):
    hits_size = len(history)
    t = n[0]
    n = n[1]
    if hits_size > 0:
        prev_x = history[hits_size - 1][1]
        prev_t = history[hits_size - 1][0]
        x = -prev_x/tau + np.sqrt(2 / tau) * n
        x *= (t - prev_t) # time scaling of the process
    else:
        x = -t*init_cond/tau + np.sqrt(2 / tau) * n
        x *= t # time scaling
    history.append([t, x])
    return history

def ou(noise_arr, tau, initial_condition=0):
    return np.array(functools.reduce(functools.partial(ou_step, initial_condition, tau),
                            noise_arr,
                            []))
