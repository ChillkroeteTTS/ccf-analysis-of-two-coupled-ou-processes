import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import shutil

from file_handling import write_simulations_to_disk
from noise import NoiseType
from stats import delayed_ou_processes_ensemble, SimulationResults

T = 1  # delay
T_cycles = 2
initial_condition = 0
R = 1000  # resolution
ensemble_runs = 700
# R = 100  # resolution
# ensemble_runs = 50

t = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles


steps = [0.1, 0.3, 0.5, 0.7, 0.9]
params_symmetric_increasing_taus = [{'e': e, 'tau1': tau, 'tau2': tau, 'noiseType': {'type': NoiseType.WHITE}} for e in steps for tau in steps] \
    + [{'e': e, 'tau1': tau, 'tau2': tau, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for e in steps for tau in steps]

params_1 = [
    # WHITE, increasing e
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}},
    {'e': 0.7, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}},

    # WHITE, increasing tau
    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.2, 'noiseType': {'type': NoiseType.WHITE}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.WHITE}},
    {'e': 0.5, 'tau1': 0.7, 'tau2': 0.7, 'noiseType': {'type': NoiseType.WHITE}},

    # RED, increasing e
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.7, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, increasing tau
    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.2, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.7, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, increasing gamma
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.2, 'gamma2': 0.2}},
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.7, 'gamma2': 0.7}},

    # RED, tau 2 increasing (smaller, equal, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.2, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.8, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, tau 1 increasing (smaller, equal, much bigger)
    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.8, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, gamma 2 increasing (smaller, equal, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.2}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.8}},

    # RED, gamma 1 increasing (smaller, equal, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.2, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.8, 'gamma2': 0.5}},
]

paramsTauVsE = \
    [{'e': i, 'tau1': 0.1, 'tau2': 0.1, 'noiseType': {'type': NoiseType.WHITE}} for i in np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.7, 'tau2': 0.7, 'noiseType': {'type': NoiseType.WHITE}} for i in np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.WHITE}} for i in np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}} for i in np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 1, 'tau2': 1, 'noiseType': {'type': NoiseType.WHITE}} for i in np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.1, 'tau2': 0.1, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for i in
     np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.7, 'tau2': 0.7, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for i in
     np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for i in
     np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for i in
     np.arange(0, 1.2, 0.2)] + \
    [{'e': i, 'tau1': 1, 'tau2': 1, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for i in
     np.arange(0, 1.2, 0.2)]


def simulate_on_params(ps):
    return ps


def get_white_noise(a):
    return a[:7]


def get_red_noise(a):
    return a[6:12]


def get_symm_increasing_gamma(a):
    return a[12:15]


def get_different_taus(a):
    return a[15:21]


def get_different_gammas(a):
    return a[21:27]


def wrapped_delayed_processes(p) -> SimulationResults:
    return delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, ensemble_runs)


def calculations(params) -> List[SimulationResults]:
    # parallelized simulation
    pool = mp.Pool(processes=12)
    return pool.map(wrapped_delayed_processes, params)


def calc_and_save():
    params = params_symmetric_increasing_taus
    name = 'params_symmetric_increasing_taus'
    start_time = time.perf_counter()

    results: List[SimulationResults] = calculations(params)

    result_path = Path.cwd() / f"results/{name}_{ensemble_runs}_{R}_{initial_condition}"
    print(f"It took {time.perf_counter() - start_time}ms to finish calculations")
    print('simulations done, write to ' + str(result_path))

    write_simulations_to_disk(result_path, results)

    print(f"It took {time.perf_counter() - start_time}ms to write output data")
    write_done = time.perf_counter()

    # res = plot_results(results, show_acf, show_ccf, show_correlation, show_different_taus, show_samples)
    print(f"It took {time.perf_counter() - write_done}ms to prepare plots")

    # plt.show()
    return results



if __name__ == '__main__':
    calc_and_save()
