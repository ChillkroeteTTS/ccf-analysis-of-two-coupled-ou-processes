import functools

import json
from io import StringIO

import numpy as np
from numbers import Number

import pandas as pd
from pandas import DataFrame, Series

from typing import TypedDict, Any, List, Dict, Tuple
from noise import white_noise, NoiseType, red_noise
from ou import ou, mixed_noise_ou


class PercentileResult(DataFrame):
    median: Series
    lower_percentile: Series
    upper_percentile: Series


class SimulationResults(DataFrame):
    p: Dict
    acf_ensemble: DataFrame  # Contains percentileResults prefixed with either ou1_ or ou2
    ccf_ensemble: PercentileResult
    ensemble: List[DataFrame]  # List of realizations. Each realization has colmuns for ou1, ou2, noise1, noise2 etc.


def normalize(ts):
    return (ts - np.mean(ts)) / np.std(ts)


def acf(ser: Series, lags: Number):
    return Series([ser.autocorr(lag) for lag in range(0, lags)])


def ccf(df: DataFrame, column1: str, column2: str, range: List[Number]) -> Series:
    return Series([df[column1].corr(df[column2].shift(lag)) for lag in range], index=range)


def group_by_index(dfs: List[DataFrame]):
    concatted: DataFrame = functools.reduce(lambda agg, df: pd.concat((agg, df)), dfs)
    return concatted.groupby(concatted.index)


def run_ou_process_realization(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition) -> DataFrame:
    """

    :param R:
    :param T_cycles:
    :param t:
    :param tau1:
    :param tau2:
    :param e:
    :param noise_type:
    :param initial_condition:
    :return:
        Dataframe containing all time series relevant for both processes as well as the processes themself
    """
    noise = white_noise if noise_type['type'] == NoiseType.WHITE else functools.partial(red_noise, noise_type['gamma1'])

    noise1 = noise(t.size)
    noise2 = noise(t.size)

    ou1 = ou(np.dstack((t, noise1))[0], tau1, initial_condition)
    ou1 = ou1[:, 2]

    [mixed_noise, ou2] = mixed_noise_ou(t, noise1, noise2, R, T_cycles, e, tau2, initial_condition)

    return DataFrame({'noise1': noise1, 'noise2': noise2, 'mixed_noise': mixed_noise, 'ou1': ou1, 'ou2': ou2})


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


def ensemble_percentiles(ensemble: List[DataFrame], fn, p) -> DataFrame:
    results = [fn(realization) for realization in ensemble]
    grouped = group_by_index(results)
    return DataFrame({
        'median': grouped.median(),
        'lower_percentile': grouped.quantile(.25),
        'upper_percentile': grouped.quantile(.75),
    })


# Simulates ensembles of both OU processes and caculates a cross correlation functions ensemble on it
def delayed_ou_processes_ensemble(R, T_cycles, t, p, initial_condition, ensemble_count) -> SimulationResults:
    tau1 = p['tau1']
    tau2 = p['tau2']
    e = p['e']
    noise_type = p['noiseType']
    ensemble: List[DataFrame] = [
        run_ou_process_realization(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition)
        for _ in
        range(0, ensemble_count)]

    print('calculating acfs for params', R, T_cycles, tau1, tau2, e, noise_type)
    acf_percentiles = acfs_for_ensemble(ensemble, p)

    print('calculating ccfs for params', R, T_cycles, tau1, tau2, e, noise_type)
    ccf_percentiles = ensemble_percentiles(ensemble, lambda df: ccf(df, 'ou2', 'ou1', range(450, 550)), p) \
        .add_prefix('ccf_')

    print('calculating ensemble percentiles for params', R, T_cycles, tau1, tau2, e, noise_type)
    grouped = group_by_index(ensemble)
    ensemble_median: DataFrame = grouped.median().add_suffix('_median')
    ensemble_lower_percentile: DataFrame = grouped.quantile(.25).add_suffix('_25p')
    ensemble_upper_percentile: DataFrame = grouped.quantile(.75).add_suffix('_75p')

    return {'p': p,
            'ensemble': ensemble_median
                .merge(ensemble_lower_percentile, left_index=True, right_index=True)
                .merge(ensemble_upper_percentile, left_index=True, right_index=True),
            'acf_ensemble': acf_percentiles,
            'ccf_ensemble': ccf_percentiles,
            'raw_ensemble': ensemble}


def acfs_for_ensemble(ensemble, p) -> DataFrame:
    lag = 100
    acf_ensemble_ou1 = ensemble_percentiles(ensemble, lambda realization: acf(realization['ou1'], lag), p)
    acf_ensemble_ou2 = ensemble_percentiles(ensemble, lambda realization: acf(realization['ou2'], lag), p)

    return acf_ensemble_ou1.add_prefix('acf_ou1_').merge(acf_ensemble_ou2.add_prefix('acf_ou2_'), left_index=True,
                                                     right_index=True)


def from_json(str: str):
    nested = json.loads(str)
    return {
        'p': nested['p'],
        'median': pd.read_csv(StringIO(nested['median'])),
        'lower_percentile': pd.read_csv(StringIO(nested['lower_percentile'])),
        'upper_percentile': pd.read_csv(StringIO(nested['upper_percentile'])),
        'ccf_median': Series(nested['ccf_median']),
        'ccf_lower_percentile': Series(nested['ccf_lower_percentile']),
        'ccf_upper_percentile': Series(nested['ccf_upper_percentile']),
        'ensemble': [pd.read_csv(StringIO(e)) for e in nested['ensemble']]
    }
