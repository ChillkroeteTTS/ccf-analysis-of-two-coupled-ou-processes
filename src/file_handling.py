import os
import json
import shutil
from pathlib import Path

import pandas as pd
from typing import List

from noise import NoiseType
from stats import SimulationResults


def write_simulations_to_disk(result_path, results):
    if os.path.exists(str(result_path)):
        print('delete existing ' + str(result_path))
        shutil.rmtree(str(result_path))
    os.makedirs(str(result_path), exist_ok=True)
    for simulation in results:
        simulation_dir = generate_simulation_dir(result_path, simulation['p'])
        os.makedirs(str(simulation_dir), exist_ok=True)

        with open(str(simulation_dir / 'p.json'), 'w+') as f:
            f.write(json.dumps(simulation['p']))

        simulation['ensemble'].to_csv(simulation_dir / 'ensemble.csv', index=False)
        simulation['acf_ensemble'].to_csv(simulation_dir / 'acf_ensemble.csv', index=True, index_label='offset')
        simulation['ccf_ensemble'].to_csv(simulation_dir / 'ccf_ensemble.csv', index=True, index_label='offset')


def generate_simulation_dir(result_path, p):
    noise_type = p["noiseType"]["type"]
    e = p["e"]
    t1 = p["tau1"]
    t2 = p["tau2"]
    gamma1 = p['noiseType']["gamma1"] if p['noiseType']['type'] == NoiseType.RED else 0
    gamma2 = p['noiseType']["gamma2"] if p['noiseType']['type'] == NoiseType.RED else 0
    return result_path / f'{noise_type}_{e}_{t1}_{t2}_{gamma1}_{gamma2}'


def load_ensemble(base_path: Path) -> List[SimulationResults]:
    return [load_simulation_result(base_path, simulation_dir) for simulation_dir in os.listdir(base_path)]


def load_simulation_result(base_path, simulation_dir):
    simulat_path = base_path / simulation_dir
    p = json.loads(open(simulat_path / 'p.json', 'r').read())
    ensemble = pd.read_csv(simulat_path / 'ensemble.csv')
    acf_ensemble = pd.read_csv(simulat_path / 'acf_ensemble.csv', index_col='offset')
    ccf_ensemble = pd.read_csv(simulat_path / 'ccf_ensemble.csv', index_col='offset')
    return {
        'p': p,
        'ensemble': ensemble,
        'acf_ensemble': acf_ensemble,
        'ccf_ensemble': ccf_ensemble,
    }
