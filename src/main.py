import os

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf, ccf

from mixedNoiseOu import mixed_noise_ou
from noise import white_noise
from ou import ou
from plotting.plotting import plt_noise, plt_ou, plt_acf

T = 1 # delay
R = 1000 # resolution
T_cycles = 2
t = np.linspace(0, T_cycles, R) # run simulation for 2 noise cycles
tau1 = 0.3
tau2 = 0.3
e = 0
initial_condition = 5

noise = white_noise

noise1 = noise(t.size)
noise2 = noise(t.size)

plt_noise(t, noise1, noise2)

ou1 = ou(np.dstack((t, noise1))[0], tau1, initial_condition)
ou1 = ou1[:, 2]

ou2 = mixed_noise_ou(t, noise1, noise2, R, T_cycles, e, tau2, initial_condition)

plt_ou(t, ou1, ou2)

lags = 1000
plt_acf(acf(ou1, nlags=lags), acf(ou2, nlags=lags))
# my_ccf = [np.correlate(ou1, np.roll(ou2, int(t))) for t in np.arange(round(R/2-5), round(R/2+5), 1)]
my_ccf = [np.correlate(ou1, np.roll(ou2, int(t))) for t in np.arange(round(R/2-1000), round(R/2+1000), 1)]

fig = plt.figure()
plt.suptitle('Cross Correlation Function')
plt.plot(my_ccf)
plt.show()

os._exit(0)
