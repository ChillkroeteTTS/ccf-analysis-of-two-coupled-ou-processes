import os

import numpy as np

from noise import NoiseType
from plotting.plotting import plt_noise, plt_ou, plt_acf, plt_time_series
from stats import delayed_ou_processes_ensemble

T = 1  # delay
R = 1000  # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
tau = 0.3
tau1 = tau
tau2 = tau
e = 0.5
initial_condition = 0
ensemble_runs = 100

params = [
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.7, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},

    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.2, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.7, 'tau2': 0.7, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
]

results = [delayed_ou_processes_ensemble(R, T_cycles, t, p['tau1'], p['tau2'], p['e'], initial_condition, ensemble_runs) for p in params]

plt_time_series(params, [[r['ccf_shifts']] for r in results], [[r['ccf']] for r in results], '', xlabel='lag', ylabel='CCF')

acf_t = np.arange(0,results[0]['acf_lags']+1)
plt_time_series(params,
                [[acf_t, acf_t] for r in results],
                [[r['acf_ou1'], r['acf_ou2']] for r in results],
                '',
                labels=['ou1', 'mixed ou'], xlabel='lag', ylabel='ACF')

os._exit(0)
