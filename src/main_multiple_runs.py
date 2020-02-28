import os
import time
import numpy as np
from noise import NoiseType
from plotting.plotting import plt_time_series, plt_2_graphs_with_same_axis, plt_samples, plt_correlation
from stats import delayed_ou_processes_ensemble
import multiprocessing as mp

# TODO: Fix acf, much higher lags, 

T = 1  # delay
R = 100  # resolution
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


def calc_and_plot(show_samples=True, show_acf=True, show_ccf=True, show_correlation=True, show_different_taus=True):
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
        results_without_asymm_params = np.concatenate((get_white_noise(results), get_red_noise(results), get_symm_increasing_gamma(results)))
        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results_without_asymm_params],
                        [[r['acf_ou1'][1:], r['acf_ou2'][1:]] for r in results_without_asymm_params],
                        '',
                        percentiles_per_run=[[[r['acf_ou1_percentiles'][0][1:], r['acf_ou1_percentiles'][1][1:]],
                                              [r['acf_ou2_percentiles'][0][1:], r['acf_ou2_percentiles'][1][1:]]] for r
                                             in results_without_asymm_params],
                        labels=['ou1', 'mixed ou'], xlabel='lag', ylabel='ACF')
        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results_without_asymm_params],
                        [[r['acf_ou1'][1:], r['acf_noise1'][1:]] for r in results_without_asymm_params],
                        '',
                        percentiles_per_run=[[[r['acf_ou1_percentiles'][0][1:], r['acf_ou1_percentiles'][1][1:]],
                                              [r['acf_noise1_percentiles'][0][1:], r['acf_noise1_percentiles'][1][1:]]]
                                             for r in results_without_asymm_params],
                        labels=['ou1', 'noise1'], xlabel='lag', ylabel='ACF')

        plt_time_series(params,
                        [[acf_t[1:], acf_t[1:]] for r in results_without_asymm_params],
                        [[r['acf_ou2'][1:], r['acf_noise1'][1:]] for r in results_without_asymm_params],
                        '',
                        percentiles_per_run=[[[r['acf_ou2_percentiles'][0][1:], r['acf_ou2_percentiles'][1][1:]],
                                              [r['acf_mixed_noise_percentiles'][0][1:],
                                               r['acf_mixed_noise_percentiles'][1][1:]]] for r in results_without_asymm_params],
                        labels=['mixed ou', 'mixed_noise'], xlabel='lag', ylabel='ACF')

    if show_ccf:
        plt_time_series(get_white_noise(params),
                        [[r['ccf_shifts']] for r in get_white_noise(results)],
                        [[r['ccf']] for r in get_white_noise(results)],
                        '',
                        percentiles_per_run=[[r['ccf_percentiles']] for r in get_white_noise(results)],
                        xlabel='lag',
                        ylabel='CCF')

        plt_time_series(get_red_noise(params),
                        [[r['ccf_shifts']] for r in get_red_noise(results)],
                        [[r['ccf']] for r in get_red_noise(results)],
                        '',
                        percentiles_per_run=[[r['ccf_percentiles']] for r in get_red_noise(results)],
                        xlabel='lag',
                        ylabel='CCF')

    if show_correlation:
        plt_correlation(results)

    if show_different_taus:
        partition = lambda a: [a[i*3:i*3 + 3] for i in [0,1]]
        plt_time_series(get_different_taus(params),
                        partition([r['ccf_shifts'] for r in get_different_taus(results)]),
                        partition([r['ccf'] for r in get_different_taus(results)]),
                        '',
                        percentiles_per_run=partition([r['ccf_percentiles'] for r in get_different_taus(results)]),
                        labels=[[r'$\tau_2$ = 0.2', r'$\tau_2$ = 0.5', r'$\tau_2$ = 0.8'], [r'$\tau_1$ = 0.2', r'$\tau_1$ = 0.5', r'$\tau_1$ = 0.8']],
                        subTitleFn=lambda params, i: f"fixed $\\tau_1$={params[i*3]['tau1']}" if i == 0 else f"fixed $\\tau_2$={params[i*3]['tau2']}",
                        xlabel='lag',
                        ylabel='CCF')

        plt_time_series(get_different_gammas(params),
                        partition([r['ccf_shifts'] for r in get_different_gammas(results)]),
                        partition([r['ccf'] for r in get_different_gammas(results)]),
                        '',
                        percentiles_per_run=partition([r['ccf_percentiles'] for r in get_different_gammas(results)]),
                        labels=[[r'$\gamma_2$ = 0.2', r'$\gamma_2$ = 0.5', r'$\gamma_2$ = 0.8'], [r'$\gamma_1$ = 0.2', r'$\gamma_1$ = 0.5', r'$\gamma_1$ = 0.8']],
                        subTitleFn=lambda params, i: f"fixed $\\gamma_1$={params[i*3]['noiseType']['gamma1']}" if i == 0 else f"fixed $\\gamma_2$={params[i*3]['noiseType']['gamma2']}",
                        xlabel='lag',
                        ylabel='CCF')

    took = time.perf_counter() - start_time
    print(f"It took {took}ms to finish calculations")
    return results


calc_and_plot(True, True, True, True, True)
