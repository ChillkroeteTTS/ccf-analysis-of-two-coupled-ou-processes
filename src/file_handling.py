import os
import json
import shutil
from pathlib import Path

import pandas as pd
from typing import List, Tuple
import numpy as np
from noise import NoiseType
from stats import SimulationResults


def write_simulations_to_disk(result_path, results, raw_ensemble_fields=[]):
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
        simulation['acf_ensemble'].to_csv(simulation_dir / 'acf_ensemble.csv', index=True, index_label='t_lag')
        simulation['ccf_ensemble'].to_csv(simulation_dir / 'ccf_ensemble.csv', index=True, index_label='t_lag')

        for f in raw_ensemble_fields:
            index = simulation['ensemble']['ou1_median'].index
            pd.DataFrame({str(k): sim[f] for k, sim in enumerate(simulation['raw_ensemble'])}, index=index).to_csv(simulation_dir / f'{f}_raw_ensemble.csv', index=True, index_label='offset')


def generate_simulation_dir(result_path, p):
    noise_type = p["noiseType"]["type"]
    e = p["e"]
    t1 = p["tau1"]
    t2 = p["tau2"]
    gamma1 = p['noiseType']["gamma1"] if p['noiseType']['type'] == NoiseType.RED else 0
    gamma2 = p['noiseType']["gamma2"] if p['noiseType']['type'] == NoiseType.RED else 0
    return result_path / f'{noise_type}_{e}_{t1}_{t2}_{gamma1}_{gamma2}'


def load_ensemble(base_path: Path) -> Tuple[List[SimulationResults], List[float] ,List[float], List[float]]:
    parse_steps_e = lambda dir: float(dir.split('_')[1])
    parse_steps_tau1 = lambda dir: float(dir.split('_')[2])
    parse_steps_tau2 = lambda dir: float(dir.split('_')[3])
    ensembles = 'params_symmetric_increasing_taus_300_700_0'.split('_')[4]
    R = 'params_symmetric_increasing_taus_300_700_0'.split('_')[5]

    meta = np.array([[load_simulation_result(base_path, simulation_dir), parse_steps_e(simulation_dir),
              parse_steps_tau1(simulation_dir), parse_steps_tau2(simulation_dir)] for simulation_dir in
             os.listdir(base_path) if not simulation_dir.startswith('.')], dtype="object")
    results = meta[:, 0]
    steps = sorted(list(set(meta[:, 1])))
    steps_tau1 = sorted(list(set(meta[:, 2])))
    steps_tau2 = sorted(list(set(meta[:, 3])))
    return results, steps, steps_tau1, steps_tau2, ensembles, R


def load_simulation_result(base_path, simulation_dir, raw_ensemble_fields=[]):
    simulat_path = base_path / simulation_dir
    p = json.loads(open(simulat_path / 'p.json', 'r').read())
    ensemble = pd.read_csv(simulat_path / 'ensemble.csv')
    acf_ensemble = pd.read_csv(simulat_path / 'acf_ensemble.csv', index_col='t_lag')
    ccf_ensemble = pd.read_csv(simulat_path / 'ccf_ensemble.csv', index_col='t_lag')
    res = {
        'p': p,
        'ensemble': ensemble,
        'acf_ensemble': acf_ensemble,
        'ccf_ensemble': ccf_ensemble,
    }
    for f in raw_ensemble_fields:
        res[f] = pd.read_csv(simulat_path / f'{f}_raw_ensemble.csv', index_col='t_lag')
    return res
