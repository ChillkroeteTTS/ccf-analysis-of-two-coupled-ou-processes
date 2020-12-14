import functools
from numbers import Number
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from noise import white_noise, NoiseType, red_noise, build_mixed_noise_fn
from ou import ou


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


def acf(ser: Series, R: float, t_lag: float):
    lags = [round(n) for n in np.linspace(0, t_lag * R, 100)]
    return Series([ser.autocorr(lag) for lag in lags], index=lags)


def ccf(df: DataFrame, column1: str, column2: str, range: List[Number]) -> Series:
    return Series([df[column1].corr(df[column2].shift(lag)) for lag in range], index=range)


def group_by_index(dfs: List[DataFrame]):
    concatted: DataFrame = functools.reduce(lambda agg, df: pd.concat((agg, df)), dfs)
    return concatted.groupby(concatted.index)


def run_ou_process_realization(R, T_cycles, T_interval, tau1, tau2, e, noise_type, initial_condition) -> DataFrame:
    """

    :param R: Resolution
    :param T_cycles: How many times the delay period is repeated
    :param T_interval: Simulation period
    :param tau1: Relaxation coefficient for first OU process
    :param tau2: Relaxation coefficient for second OU process
    :param e: Combination parameter for mixed noise $\epsilon \in [0, 1]$
    :param noise_type: NoiseType.WHITE or NoiseType.RED noise
    :param initial_condition: Initial condition for the process
    :return:
        Dataframe containing all time series relevant for both processes as well as the processes themself
    """
    noise_fn = white_noise if noise_type['type'] == NoiseType.WHITE else functools.partial(red_noise, noise_type['gamma1'])

    res_ou1 = ou(T_interval, tau1, noise_fn, initial_condition)
    ou1 = res_ou1[:, 2]
    noise1 = res_ou1[:, 1]

    mixed_noise_fn = build_mixed_noise_fn(T_interval, noise_fn, noise1, R, T_cycles, e)

    res_ou2 = ou(T_interval, tau1, mixed_noise_fn, initial_condition)
    ou2 = res_ou2[:, 2]
    mixed_noise = res_ou2[:, 1]

    return DataFrame({'noise1': noise1, 'mixed_noise': mixed_noise, 'ou1': ou1, 'ou2': ou2})


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


def ensemble_percentiles(ensemble: List[DataFrame], fn) -> DataFrame:
    results = [fn(realization) for realization in ensemble]
    grouped = group_by_index(results)
    return DataFrame({
        'median': grouped.median(),
        'lower_percentile': grouped.quantile(.25),
        'upper_percentile': grouped.quantile(.75),
    })


def delayed_ou_processes_ensemble(R: float,
                                  T_cycles: int,
                                  t_interval: List[float],
                                  p: dict,
                                  initial_condition: float,
                                  ensemble_count: int) -> SimulationResults:
    """
    Simulates ensembles of both OU processes and caculates a cross correlation functions ensemble on it
    :param R: Resolution
    :param T_cycles: How many times the delay period is repeated
    :param t_interval: Simulation period
    :param p: Parameter set
    :param initial_condition: Initial condition of the process
    :param ensemble_count: Number of process realizations
    :return: Simulation results containing the parameter set, the simulated median, 25p and 75p percentile of the
    simulated ensemble, the acf ensemble percentiles, the ccf percentiles and the unaggregated ensemble simulation
    """
    tau1 = p['tau1']
    tau2 = p['tau2']
    e = p['e']
    noise_type = p['noiseType']
    ensemble: List[DataFrame] = [
        run_ou_process_realization(R, T_cycles, t_interval, tau1, tau2, e, noise_type, initial_condition)
        for _ in
        range(0, ensemble_count)]

    print('calculating acfs for params', R, T_cycles, tau1, tau2, e, noise_type)
    acf_percentiles = acfs_for_ensemble(R, ensemble, p)

    print('calculating ccfs for params', R, T_cycles, tau1, tau2, e, noise_type)
    x = list(range(420, 581))
    ccf_percentiles = ensemble_percentiles(ensemble, lambda df: ccf(df, 'ou2', 'ou1', x)) \
        .add_prefix('ccf_')

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


def acfs_for_ensemble(R, ensemble, p) -> DataFrame:
    t_lag = 0.2
    acf_ensemble_ou1 = ensemble_percentiles(ensemble, lambda realization: acf(realization['ou1'], R, t_lag))
    acf_ensemble_ou2 = ensemble_percentiles(ensemble, lambda realization: acf(realization['ou2'], R, t_lag))

    return acf_ensemble_ou1.add_prefix('acf_ou1_').merge(acf_ensemble_ou2.add_prefix('acf_ou2_'), left_index=True,
                                                     right_index=True)
