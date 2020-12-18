from pathlib import Path

from pandas import np

from file_handling import write_simulations_to_disk
from noise import NoiseType
from stats import delayed_ou_processes_ensemble

T = 1  # delay
T_cycles = 2
initial_condition = 0
R = 500  # resolution
ensemble_runs = 1000
# R = 100
# ensemble_runs = 50

T_interval = np.linspace(0, T_cycles, R)  # run simulation for 2 noise cycles
p = {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.RED, 'gamma1': 0.5, 'gamma2': 0.5}}
p2 = {'e': 0.2, 'tau1': 0.3, 'tau2': 0.3, 'noiseType': {'type': NoiseType.WHITE}}
name = 'red_noise_test'

results = delayed_ou_processes_ensemble(R, T_cycles, T_interval, p, initial_condition, ensemble_runs)

result_path = Path.cwd() / f"results/{name}_{ensemble_runs}_{R}_{initial_condition}"

write_simulations_to_disk(result_path, [results], raw_ensemble_fields=['noise1', 'mixed_noise'])
print('simulations done, write to ' + str(result_path))
