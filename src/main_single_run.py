import os

import numpy as np
import matplotlib.pyplot as plt
from plotting.plotting import plt_noise, plt_ou, plt_acf
from stats import delayed_ou_processes_ensemble

T = 1 # delay
R = 1000 # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R) # run simulation for 2 noise cycles
tau = 0.3
tau1 = tau
tau2 = tau
e = 0.5
initial_condition = 0

res = delayed_ou_processes_ensemble(R, T_cycles, t, tau1, tau2, e, initial_condition)
noise1 = res['noise1']
noise2 = res['noise2']
ou1 = res['ou1']
ou2 = res['ou2']
acf_ou1 = res['acf_ou1']
acf_ou2 = res['acf_ou2']
ccf = res['ccf']
ccf_shifts = res['ccf_shifts']

plt_noise(t, noise1, noise2)
plt_ou(t, ou1, ou2)
plt_acf(acf_ou1, acf_ou2)

fig = plt.figure()
plt.suptitle('Cross Correlation Function')
plt.plot(ccf_shifts, ccf)
plt.xlabel('shift')
plt.ylabel('correlation')
plt.show()

os._exit(0)
