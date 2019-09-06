import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf

from noise import white_noise
from ou import ou
from plotting.plotting import plt_noise, plt_ou

T = 1 # delay
R = 1000 # resolution
t = np.linspace(0, 2, R) # run simulation for 2 noise cycles
noise = white_noise

noise1 = noise(t.size)
noise2 = noise(t.size)

plt_noise(t, noise1, noise2)

ou1 = ou(noise1, 2, 0)
# print(ou1)
plt_ou(t, ou1)
print(np.average(ou1))

fig, axs = plt.subplots(2, 1, sharey=True)
axs[0].plot(acf(noise1))
axs[1].plot(acf(ou1))
plt.show()

