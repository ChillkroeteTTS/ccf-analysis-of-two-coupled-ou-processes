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
        x = init_cond
    history.append([t, n,  x])
    return history

def ou(noise_arr, tau, initial_condition=0):
    return np.array(functools.reduce(functools.partial(ou_step, initial_condition, tau),
                            noise_arr,
                            []))


def mixed_noise_ou(t, noise1, noise2, R, T_cycles, e, tau2, inital_condition):
    # Zeitpunkt bei dem der mixed noise Prozess aufgrund der Verzögerung starten kann (t1 = 1)
    i_t_1 = int(np.ceil(R / T_cycles))
    # Zeitbereich in dem der mn prozess läuft
    t_mixed = t[i_t_1:]
    pow_e = np.power(e, 2.0)

    # Noise im Zeitinterval t0 - t1
    noise_t0_t1 = noise1[:i_t_1]
    # Noise im Zeitinterval t1 - ende (t2)
    noise_t1_t2 = noise2[i_t_1:]

    mixed_noise = noise_t0_t1 * np.sqrt(pow_e) + noise_t1_t2 * np.sqrt(1 - pow_e)

    # [[t1, mn0], [t2, mn1] ...]
    partitioned_noise = np.dstack((t_mixed, mixed_noise))[0]

    ou2 = ou(partitioned_noise, tau2, inital_condition)

    # zero pre padding for same dimensions of ou1 and ou2
    return np.concatenate((np.full(i_t_1, 0), ou2[:, 2]))
