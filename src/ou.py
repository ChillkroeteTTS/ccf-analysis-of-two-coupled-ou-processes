import functools
from typing import List

import numpy as np


def ou_step(init_cond, tau, history: List, n):
    hits_size = len(history)
    t = n[0]
    n = n[1]
    if hits_size > 0:
        prev_x = history[hits_size - 1][2]
        prev_n = history[hits_size - 1][1]
        prev_t = history[hits_size - 1][0]
        delta_n = (n - prev_n)
        delta_t = (t - prev_t)
        x = prev_x + delta_t * (-prev_x / tau) + np.sqrt(2 / tau) * delta_n
    else:
        x = init_cond - t*init_cond/tau + np.sqrt(2 / tau) * n
    history.append([t, n,  x])
    return history

def ou(noise_arr, tau, initial_condition=0):
    return np.array(functools.reduce(functools.partial(ou_step, initial_condition, tau),
                            noise_arr,
                            []))
