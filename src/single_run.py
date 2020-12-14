from pandas import np

from noise import NoiseType
from stats import delayed_ou_processes_ensemble

T = 1  # delay
T_cycles = 2
initial_condition = 0
R = 100  # resolution
ensemble_runs = 50

T_interval = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
p = {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}}

delayed_ou_processes_ensemble(R, T_cycles, T_interval, p, initial_condition, ensemble_runs)
