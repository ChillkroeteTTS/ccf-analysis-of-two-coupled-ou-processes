import functools

import json
from io import StringIO

import numpy as np
from numbers import Number

import pandas as pd
from pandas import DataFrame, Series

from typing import TypedDict, Any, List, Dict
from noise import white_noise, NoiseType, red_noise
from ou import ou, mixed_noise_ou

class SimulationResults(TypedDict):
    p: Dict
    median: DataFrame
    lower_percentile: DataFrame
    upper_percentile: DataFrame
    ensemble: List[DataFrame]

class PercentileResult(TypedDict):
    p: Dict
    median: DataFrame
    lower_percentile: DataFrame
    upper_percentile: DataFrame

def normalize(ts):
    return (ts - np.mean(ts)) / np.std(ts)


def acf(ser: Series, lags: Number):
    return Series([ser.autocorr(lag) for lag in range(0, lags)])


def ccf(df: DataFrame, column1: str, column2: str, range: List[Number]):
    return Series([df[column1].corr(df[column2].shift(lag)) for lag in range], index=range)


def group_by_index(dfs: List[DataFrame]):
    concatted: DataFrame = functools.reduce(lambda agg, df: pd.concat((agg, df)), dfs)
    return concatted.groupby(concatted.index)


def delayed_ou_processes(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition):
    noise = white_noise if noise_type['type'] == NoiseType.WHITE else functools.partial(red_noise, noise_type['gamma1'])

    noise1 = noise(t.size)
    noise2 = noise(t.size)

    ou1 = ou(np.dstack((t, noise1))[0], tau1, initial_condition)
    ou1 = ou1[:, 2]

    [mixed_noise, ou2] = mixed_noise_ou(t, noise1, noise2, R, T_cycles, e, tau2, initial_condition)

    df_base_data = DataFrame({'noise1': noise1, 'noise2': noise2, 'mixed_noise': mixed_noise, 'ou1': ou1, 'ou2': ou2})
    return df_base_data


# Returns the index of the value in the time series where the integral of the curve reaches 50%
def i_50(ts):
    total = np.sum(ts)

    def reduce_to_iqr(agg, v):
        [moving_sum, i_50] = agg
        [i, y] = v

        new_total = moving_sum + y
        return [new_total, i if i_50 == 0 and new_total > (0.5 * total) else i_50]

    [_, i] = functools.reduce(reduce_to_iqr, enumerate(ts), [0, 0])
    return i

def ensemble_percentiles(ensemble: List[DataFrame], fn, p) -> PercentileResult:
    results = [fn(realization) for realization in ensemble]
    grouped = group_by_index(results)
    return {
        'median': grouped.median(),
        'p': p,
        'lower_percentile': grouped.quantile(.25),
        'upper_percentile': grouped.quantile(.75),
    }

def delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, ensemble_count):
    tau1 = p['tau1']
    tau2 = p['tau2']
    e = p['e']
    noise_type = p['noiseType']
    ensemble_runs: List[DataFrame] = [delayed_ou_processes(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition)
                                      for _ in
                                      range(0, ensemble_count)]

    return {'p': p,
            'median': group_by_index(ensemble_runs).median(),
            'lower_percentile': group_by_index(ensemble_runs).quantile(.25),
            'upper_percentile': group_by_index(ensemble_runs).quantile(.75),
            'ensemble': ensemble_runs}


def to_json(res: SimulationResults):
    return json.dumps({
        'p': res['p'],
        'median': res['median'].to_csv(),
        'lower_percentile': res['lower_percentile'].to_csv(),
        'upper_percentile': res['upper_percentile'].to_csv(),
        'ensemble': [e.to_csv() for e in res['ensemble']]}
    )

def from_json(str: str):
    nested = json.loads(str)
    return {
        'p': nested['p'],
        'median': pd.read_csv(StringIO(nested['median'])),
        'lower_percentile': pd.read_csv(StringIO(nested['lower_percentile'])),
        'upper_percentile': pd.read_csv(StringIO(nested['upper_percentile'])),
        'ensemble': [pd.read_csv(StringIO(e)) for e in nested['ensemble']]
    }
