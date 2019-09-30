import numpy as np


# t - Zeit
# e - [0, 1] gewichtet die Noise Anteile
from ou import ou


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
