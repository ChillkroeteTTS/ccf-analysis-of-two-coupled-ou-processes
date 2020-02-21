import os
import time

import numpy as np

from noise import NoiseType
from plotting.plotting import plt_time_series, plt_2_graphs_with_same_axis, plt_samples, plt_correlation
from stats import delayed_ou_processes_ensemble
import multiprocessing as mp

T = 1  # delay
R = 500  # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
tau = 0.3
tau1 = tau
tau2 = tau
e = 0.5
initial_condition = 0
ensemble_runs = 100

params = [
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
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.2, 'gamma2': 0.2}},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.7, 'gamma2': 0.7}},
]


def wrapped_delayed_processes(p):
    return delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, ensemble_runs)


def calculations():
    # parallelized simulation
    pool = mp.Pool(processes=8)
    return pool.map(wrapped_delayed_processes, params)


def calc_and_plot(show_samples=True, show_acf=True, show_ccf=True):
    start_time = time.perf_counter()
    results = calculations()

    print('noise2 mean')
    print(results[0]['noise2_mean'])
    print('mixed mean')
    print(results[0]['mixed_mean'])

    if show_samples:
        plt_samples(t, round(R / T_cycles), results)

    acf_t = np.arange(0, results[0]['acf_lags'] + 1)

    if show_acf:
        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results],
                        [[r['acf_ou1'][1:], r['acf_ou2'][1:]] for r in results],
                        '',
                        percentiles_per_run=[[[r['acf_ou1_percentiles'][0][1:], r['acf_ou1_percentiles'][1][1:]],
                                              [r['acf_ou2_percentiles'][0][1:], r['acf_ou2_percentiles'][1][1:]]] for r in results],
                        labels=['ou1', 'mixed ou'], xlabel='lag', ylabel='ACF')
        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results],
                        [[r['acf_ou1'][1:], r['acf_noise1'][1:]] for r in results],
                        '',
                        percentiles_per_run=[[[r['acf_ou1_percentiles'][0][1:], r['acf_ou1_percentiles'][1][1:]],
                                              [r['acf_noise1_percentiles'][0][1:], r['acf_noise1_percentiles'][1][1:]]] for r in results],
                        labels=['ou1', 'noise1'], xlabel='lag', ylabel='ACF')

        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results],
                        [[r['acf_ou2'][1:], r['acf_noise1'][1:]] for r in results],
                        '',
                        percentiles_per_run=[[[r['acf_ou2_percentiles'][0][1:], r['acf_ou2_percentiles'][1][1:]],
                                              [r['acf_mixed_noise_percentiles'][0][1:], r['acf_mixed_noise_percentiles'][1][1:]]] for r in results],
                        labels=['mixed ou', 'mixed_noise'], xlabel='lag', ylabel='ACF')

    if show_ccf:
        plt_time_series(params,
                        [[r['ccf_shifts']] for r in results],
                        [[r['ccf']] for r in results],
                        '',
                        percentiles_per_run=[[r['ccf_percentiles']] for r in results],
                        xlabel='lag',
                        ylabel='CCF')

    plt_correlation(results)


    took = time.perf_counter() - start_time
    print(f"It took {took}ms to finish calculations")
    return results

calc_and_plot()
