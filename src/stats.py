import numpy as np


def normalize(ts):
    return (ts - np.mean(ts)) / np.std(ts)


def normalized_correlation(ts1, ts2, shifts):
    # normalization according to https://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize/5639626#5639626
    normalized_ts1 = normalize(ts1) / len(ts1)
    normalized_ts2 = normalize(ts2)
    return [np.correlate(normalized_ts1, np.roll(normalized_ts2, int(shift)))
            for shift in shifts]