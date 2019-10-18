import functools

import numpy as np

from statsmodels.tsa.stattools import acf

from noise import white_noise, NoiseType, red_noise
from ou import ou, mixed_noise_ou


def normalize(ts):
    return (ts - np.mean(ts)) / np.std(ts)


def normalized_correlation(ts1, ts2, shifts):
    # normalization according to https://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize/5639626#5639626
    normalized_ts1 = normalize(ts1) / len(ts1)
    normalized_ts2 = normalize(ts2)
    return [np.correlate(normalized_ts1, np.roll(normalized_ts2, int(shift)))
            for shift in shifts]
    # return [np.correlate(ts1, np.roll(ts2, int(shift)))
    #         for shift in shifts] * (1/np.sqrt(np.var(ts1)*np.var(ts2)))

def delayed_ou_processes(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition):
    noise = white_noise if noise_type['type'] == NoiseType.WHITE else functools.partial(red_noise, noise_type['gamma1'])

    noise1 = noise(t.size)
    noise2 = noise(t.size)

    ou1 = ou(np.dstack((t, noise1))[0], tau1, initial_condition)
    ou1 = ou1[:, 2]

    [mixed_noise, ou2] = mixed_noise_ou(t, noise1, noise2, R, T_cycles, e, tau2, initial_condition)

    lags = 20
    acf_ou1 = acf(ou1, nlags=lags, fft=False)
    acf_ou2 = acf(ou2, nlags=lags, fft=False)

    w = 15
    ccf_shifts = np.arange(round(R / 2 - w), round(R / 2 + w), 1)
    ccf = normalized_correlation(ou1, ou2, ccf_shifts)

    return {
        'noise1': noise1,
        'noise2': mixed_noise,
        'noise2_mean': np.mean(noise2),
        'mixed_mean': np.mean(mixed_noise),
        'ou1': ou1,
        'ou2': ou2,
        'acf_lags': lags,
        'acf_ou1': acf_ou1,
        'acf_ou2': acf_ou2,
        'ccf_shifts': ccf_shifts,
        'ccf': ccf,
    }


def delayed_ou_processes_ensemble(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition, ensemble_count):
    runs = [delayed_ou_processes(R, T_cycles, t, tau1, tau2, e, noise_type, initial_condition) for _ in range(0, ensemble_count)]

    average_ensemble = lambda e: np.median(e, axis=0)

    return {
        'noise1': np.array([run['noise1'] for run in runs]),
        'noise2': (np.array([run['noise2'] for run in runs])),
        'noise2_mean': average_ensemble(np.array([run['noise2_mean'] for run in runs])),
        'mixed_mean': average_ensemble(np.array([run['mixed_mean'] for run in runs])),
        'ou1': np.array([run['ou1'] for run in runs]),
        'ou2': np.array([run['ou2'] for run in runs]),
        'acf_ou1': average_ensemble(np.array([run['acf_ou1'] for run in runs])),
        'acf_ou2': average_ensemble(np.array([run['acf_ou2'] for run in runs])),
        'acf_lags': runs[0]['acf_lags'],
        'ccf_shifts': runs[0]['ccf_shifts'],
        'ccf': average_ensemble(np.array([run['ccf'] for run in runs])),
    }
