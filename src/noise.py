import numpy as np
from typing import Callable, List


class NoiseType:
    WHITE = 'white noise'
    RED = 'red noise'

def white_noise(n: int, delta_t: float) -> List[float]:
    return list(np.random.normal(loc=0, scale=np.sqrt(delta_t), size=n))


def red_noise(r: float, n: int, delta_t: float) -> List[float]:
    noise = white_noise(n, delta_t)

    for i in range(1, len(noise)):
        noise[i] = r * noise[i-1] + np.sqrt(1 - np.power(r, 2)) * noise[i]

    return noise


def build_mixed_noise_fn(t_interval: List[float],
                         noiseFn: Callable[[int ,float], float],
                         noise1,
                         R: int,
                         T_cycles: int,
                         e: float) -> Callable[[int, float], List[float]]:
    """
    Returns a function which produces noise 1 until the delay time T is reached and a combination of noise 1 and noise 2 afterwards.
    # Example with T = 1, and T_cycles = 2 -> T_total = 2
    # t_0--------t1-------t_total
    # |--noise1--|--mixed noise--|
    :param t_interval: Simulation period
    :param noiseFn: Function creating noise increments
    :param noise1: Noise increments of first OU process
    :param R: Resolution
    :param T_cycles: How many times the delay period is repeated
    :param e: $\epsilon$ controls the influence of noise1 on the resulting noise
    :return: Function which produces noise for the second OU process
    """
    def mixed_noise_fn(n: int, delta_t: float):
        # t1 = time from which on noise 1 and noise 2 can be combined because t = T
        i_t_1 = int(np.ceil(R / T_cycles))

        # Zeitbereich in dem der mn prozess läuft
        t_mixed = t_interval[i_t_1:]
        pow_e = np.power(e, 2.0)

        noise2 = noiseFn(n, delta_t)

        # Noise 1 in interval t0 - t1
        noise_t0_t1 = noise1[:i_t_1]

        # Noise 2 in interval t1 - end (t2)
        noise_t1_t2 = noise2[i_t_1:]

        # combination of noise 1 and noise 2 in interval t1 - t_total
        mixed_noise = noise_t0_t1 * np.sqrt(pow_e) + np.array(noise_t1_t2) * np.sqrt(1 - pow_e)

        # concatenation of noise 1 in interval t_0 - t1 and the mixed noise in interval t1 - t_total
        return list(np.concatenate((noise2[:i_t_1], mixed_noise)))

    return mixed_noise_fn
