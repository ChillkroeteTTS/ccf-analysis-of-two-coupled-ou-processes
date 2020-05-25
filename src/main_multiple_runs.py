import json
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from noise import NoiseType
from plotting.plotting import plt_time_series, plt_2_graphs_with_same_axis, plt_correlation, plt_sample_from_ensemble, \
    plot_multiple_percentiles
from stats import delayed_ou_processes_ensemble, SimulationResults, to_json, from_json, ensemble_percentiles, \
    PercentileResult, acf, ccf
import multiprocessing as mp

T = 1  # delay
R = 1000  # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
initial_condition = 0
ensemble_runs = 50

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
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.2, 'gamma2': 0.2}},
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.7, 'gamma2': 0.7}},

    # RED, tau 2 increasing (smaller, slightly bigger, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.2, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.8, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, tau 1 increasing (smaller, slightly bigger, much bigger)
    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.8, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},

    # RED, gamma 2 increasing (smaller, slightly bigger, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.2}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.8}},

    # RED, gamma 1 increasing (smaller, slightly bigger, much bigger)
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.2, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.8, 'gamma2': 0.5}},
]

def simulate_on_params(ps):
    return ps

def get_white_noise(a):
    return a[:6]

def get_red_noise(a):
    return a[6:12]

def get_symm_increasing_gamma(a):
    return a[12:15]

def get_different_taus(a):
    return a[15:21]

def get_different_gammas(a):
    return a[21:27]


def wrapped_delayed_processes(p):
    return delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, ensemble_runs)


def calculations():
    # parallelized simulation
    pool = mp.Pool(processes=8)
    return pool.map(wrapped_delayed_processes, simulate_on_params(params))


def plot_results(results: List[SimulationResults], show_acf, show_ccf, show_correlation, show_different_taus, show_samples):
    if show_samples:
        plt_sample_from_ensemble(t, round(R / T_cycles), results[0]['p'], results[0]['ensemble'])

    if show_acf:
        # acf_t = np.arange(0, results[0]['acf_lags'] + 1)
        results_without_asymm_params = [res for res in results if res['p']['tau1'] == res['p']['tau2']]
        lag = 100
        percentile_of_results = [[ensemble_percentiles(res['ensemble'], lambda df: acf(df['ou1'], lag), res['p']),
                                  ensemble_percentiles(res['ensemble'], lambda df: acf(df['ou2'], lag), res['p'])
                                  ]
                                 for res in results_without_asymm_params]
        plot_multiple_percentiles(percentile_of_results, 'lag', 'autocov(ou)', labels=['ou1', 'ou2'], title='Percentiles of Autocorrellation Functions')

    if show_ccf:
        white_noise_res = [res for res in results if res['p']['noiseType']['type'] == NoiseType.RED]
        percentile_of_results = [[ensemble_percentiles(res['ensemble'], lambda df: ccf(df, 'ou2', 'ou1', range(400, 600)), res['p'])]
                                 for res in white_noise_res]
        plot_multiple_percentiles(percentile_of_results, 'lag', 'autocov(ou)', labels=['ccf(ou1, ou2)'], title='Percentiles of Cross Correlation Functions')

    if show_correlation:
        plt_correlation(results)
    if show_different_taus:
        partition = lambda a: [a[i * 3:i * 3 + 3] for i in [0, 1]]
        plt_time_series(get_different_taus(params),
                        partition([r['ccf_shifts'] for r in get_different_taus(results)]),
                        partition([r['ccf'] for r in get_different_taus(results)]),
                        '',
                        percentiles_per_run=partition([r['ccf_percentiles'] for r in get_different_taus(results)]),
                        labels=[[r'$\tau_2$ = 0.2', r'$\tau_2$ = 0.5', r'$\tau_2$ = 0.8'],
                                [r'$\tau_1$ = 0.2', r'$\tau_1$ = 0.5', r'$\tau_1$ = 0.8']],
                        subTitleFn=lambda params,
                                          i: f"fixed $\\tau_1$={params[i * 3]['tau1']}" if i == 0 else f"fixed $\\tau_2$={params[i * 3]['tau2']}",
                        xlabel='lag',
                        ylabel='CCF')

        plt_time_series(get_different_gammas(params),
                        partition([r['ccf_shifts'] for r in get_different_gammas(results)]),
                        partition([r['ccf'] for r in get_different_gammas(results)]),
                        '',
                        percentiles_per_run=partition([r['ccf_percentiles'] for r in get_different_gammas(results)]),
                        labels=[[r'$\gamma_2$ = 0.2', r'$\gamma_2$ = 0.5', r'$\gamma_2$ = 0.8'],
                                [r'$\gamma_1$ = 0.2', r'$\gamma_1$ = 0.5', r'$\gamma_1$ = 0.8']],
                        subTitleFn=lambda params,
                                          i: f"fixed $\\gamma_1$={params[i * 3]['noiseType']['gamma1']}" if i == 0 else f"fixed $\\gamma_2$={params[i * 3]['noiseType']['gamma2']}",
                        xlabel='lag',
                        ylabel='CCF')

    plt.show()
    return results


def calc_and_plot(show_samples=True, show_acf=True, show_ccf=True, show_correlation=True, show_different_taus=True):
    result_path = Path.cwd() / f"results/{ensemble_runs}_{R}_{initial_condition}"
    start_time = time.perf_counter()
    results: List[SimulationResults] = calculations()
    print(f"It took {time.perf_counter() - start_time}ms to finish calculations")
    print('simulations done, write to ' + str(result_path))

    for i, res in enumerate(results):
        full_result_path = result_path / f'{i}_{res["p"]["noiseType"]["type"]}_{res["p"]["e"]}_{res["p"]["tau1"]}_{res["p"]["tau2"]}.json'
        with open(full_result_path, 'w+') as f:
            jsonStr = to_json(res)
            f.write(jsonStr)
            f.close()
    print(f"It took {time.perf_counter() - start_time}ms to write output data")

    return plot_results(results, show_acf, show_ccf, show_correlation, show_different_taus, show_samples)

def load_and_plot(base_path: Path, show_samples=True, show_acf=True, show_ccf=True, show_correlation=True, show_different_taus=True):
    start_time = time.perf_counter()
    results: List[SimulationResults] = [from_json(open(base_path / path, 'r').read()) for path in os.listdir(base_path)]
    print(f"It took {time.perf_counter() - start_time}ms to reload calculations")

    l = plot_results(results, show_acf, show_ccf, show_correlation, show_different_taus, show_samples)
    print(f"It took {time.perf_counter() - start_time}ms to plot everything")
    return l

if __name__ == '__main__':
    load_and_plot(Path.cwd() / 'results' / '50_1000_0', False, False, True, False, False)
    # calc_and_plot(True, False, False, False, False)
