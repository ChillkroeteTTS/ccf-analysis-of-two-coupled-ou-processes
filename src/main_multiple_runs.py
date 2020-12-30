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
T_total = T * T_cycles
initial_condition = 0
R = 500  # resolution
ensemble_runs = 1000
# R = 100  # resolution
# ensemble_runs = 50

t_interval = np.linspace(0, T_total, R)  # run simulation for 2 noise cycles


steps_e = np.linspace(0.05, 0.95, 7)
steps_tau = np.linspace(0.05, 0.95, 7)
steps_gamma = np.linspace(0.05, 0.95, 7)
params_symmetric_increasing_taus = [{'e': e, 'tau1': tau, 'tau2': tau, 'noiseType': {'type': NoiseType.WHITE}} for e in steps_e for tau in steps_tau] \
    + [{'e': e, 'tau1': tau, 'tau2': tau, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}} for e in steps_e for tau in steps_tau]

params_asymetric_increasing_taus =[{'e': 0.5, 'tau1': tau1, 'tau2': tau2, 'noiseType': {'type': NoiseType.WHITE}} for tau1 in steps_tau for tau2 in steps_tau]
params_asymetric_increasing_gammas =[{'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': gamma1, 'gamma2': gamma2}} for gamma1 in steps_gamma for gamma2 in steps_gamma]


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
    return delayed_ou_processes_ensemble(T_total, R, T_cycles, t_interval, p, initial_condition, ensemble_runs)


def calculations(params) -> List[SimulationResults]:
    # parallelized simulation
    pool = mp.Pool(processes=12)
    return pool.map(wrapped_delayed_processes, params)


def calc_and_save():
    params = params_asymetric_increasing_gammas
    name = 'params_asymetric_increasing_gammas'
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
