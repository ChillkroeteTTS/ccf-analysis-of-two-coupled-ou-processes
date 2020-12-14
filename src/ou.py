import functools
from typing import List, Callable, Tuple

import numpy as np


def euler_maruyama_method_step(init_cond: float,
                               drift: Callable[[float], float],
                               diffusion: Callable[[float], float],
                               history: List,
                               t_and_w: Tuple[float, float]):
    """

    :param init_cond: Initial condition of the process
    :param drift: Drift term
    :param diffusion: Diffusion term
    :param history: Collection of tuples described in return type
    :param t: T current time step t
    :param delta_w_t: random increment $\Delta w_t = w_t - w_{t-1}$
    :return: A tuple of time step t, random increment value $\Delta w_t$ and process result x(t) = x
    """
    t, delta_w_t = t_and_w
    hist_length = len(history)
    is_initial_step = hist_length > 0
    if is_initial_step:
        prev_x = history[-1][2]
        prev_t = history[-1][0]
        delta_t = (t - prev_t)
        x = prev_x + delta_t * drift(prev_x) + diffusion(prev_x) * delta_w_t
    else:
        x = init_cond
    history.append([t, delta_w_t, x])
    return history


def euler_maruyama_method(initial_condition,
                          t_interval,
                          drift: Callable[[float], float],
                          diffusion: Callable[[float], float],
                          generate_random_increments: Callable[[float, float], List[float]]):
    """
    Estimates a numerical solution to the non-linear SDE defined by drift term, diffusion term and
    :param init_cond: Initial condition of the process
    :param t_interval: Array containing timesteps in interval t_i \in [0, T_total]
    :param drift: Drift term
    :param diffusion: Diffusion term
    :param generate_random_increments: Function taking the amount of increments and $\Delta t$ as input and returns  a list of increments
    :return: A collection of tuples of time step t, noise value n(t) = n and process result x(t) = x
    """
    delta_t = t_interval[1] - t_interval[0]
    random_increments = generate_random_increments(len(t_interval), delta_t)
    return np.array(functools.reduce(functools.partial(euler_maruyama_method_step,
                                                       initial_condition, drift, diffusion),
                                     zip(t_interval, random_increments),
                                     []))


def ou(t_interval, tau, generate_noise_increments: Callable[[float, float], List[float]], initial_condition=0):
    return euler_maruyama_method(initial_condition, t_interval, lambda prev_x: (-prev_x / tau), lambda prev_x: np.sqrt(2 / tau), generate_noise_increments)
