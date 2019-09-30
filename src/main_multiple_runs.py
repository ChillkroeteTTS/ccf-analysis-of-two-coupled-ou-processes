import os
from enum import Enum
from math import floor

import numpy as np
import matplotlib.pyplot as plt

from noise import NoiseType
from plotting.plotting import plt_noise, plt_ou, plt_acf
from stats import delayed_ou_processes

T = 1  # delay
R = 1000  # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
tau = 0.3
tau1 = tau
tau2 = tau
e = 0.5
initial_condition = 0

params = [
    {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.7, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},

    {'e': 0.5, 'tau1': 0.2, 'tau2': 0.2, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.5, 'tau2': 0.5, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
    {'e': 0.5, 'tau1': 0.7, 'tau2': 0.7, 'noiseType': NoiseType.WHITE, 'gamma1': 0.5, 'gamma2': 0.5},
]

results = [delayed_ou_processes(R, T_cycles, t, p['tau1'], p['tau2'], p['e'], initial_condition) for p in params]

cols = 3
rows = int(np.ceil(len(results)/cols))
print(str(cols) + ' cols')
print(str(rows) + ' rows')
fig, axs = plt.subplots(rows, cols, sharey=True)

fig.suptitle('CCF')
for i, res in enumerate(results):
    r = int(floor(i / cols))
    c = int(i % cols)
    ccf = res['ccf']
    ccf_shifts = res['ccf_shifts']

    axs[r][c].plot(ccf_shifts, ccf, label=f"e: {params[i]['e']}, tau: {params[i]['tau1']}")
    axs[r][c].legend(loc="upper right")
    axs[r][c].set_xlabel('shift')
    axs[r][c].set_ylabel('correlation')

plt.show()

# res = delayed_ou_processes(R, T_cycles, t, tau1, tau2, e, initial_condition)
# noise1 = res['noise1']
# noise2 = res['noise2']
# ou1 = res['ou1']
# ou2 = res['ou2']
# acf_ou1 = res['acf_ou1']
# acf_ou2 = res['acf_ou2']
# ccf = res['ccf']
# ccf_shifts = res['ccf_shifts']
#
# plt_noise(t, noise1, noise2)
# plt_ou(t, ou1, ou2)
# plt_acf(acf_ou1, acf_ou2)
#
# fig = plt.figure()
# plt.suptitle('Cross Correlation Function')
# plt.plot(ccf_shifts, ccf)
# plt.xlabel('shift')
# plt.ylabel('correlation')
# plt.show()

os._exit(0)
