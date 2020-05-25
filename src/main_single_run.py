import os

import numpy as np
import matplotlib.pyplot as plt

from noise import NoiseType
from plotting.plotting import plt_sample_from_ensemble
from stats import delayed_ou_processes_ensemble, to_json, from_json

def single_run():

    T = 1 # delay
    R = 1000 # resolution
    T_cycles = 2
    t = np.linspace(0, T_cycles, R) # run simulation for 2 noise cycles
    tau = 0.3
    tau1 = tau
    tau2 = tau
    e = 0.5
    initial_condition = 0
    p = {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}}

    res = delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, 20)
    open('./results/single_run.json', 'w+').write(to_json(res))
    from_json(''.join(open('results/single_run.json', 'r').readlines()))
    plt_sample_from_ensemble(t, int(R / T_cycles), p, res['ensemble'])

    plt.show()

if __name__ == '__main__':
    single_run()
