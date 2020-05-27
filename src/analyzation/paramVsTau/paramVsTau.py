from pathlib import Path
import matplotlib.pyplot as plt
import time

import numpy as np
from pandas import Series

from main_multiple_runs import load_results
from noise import NoiseType
from plotting.plotting import plot_heatmap, gen_2d_data


def fit_max_e_corr(results):
    data = sorted([res for res in results if
              res['p']['noiseType']['type'] == NoiseType.WHITE and res['p']['tau1'] >= 0.49 and res['p'][
                  'tau1'] <= 0.51], key=lambda res: res['p']['e'])
    x = [res['p']['e'] for res in data]
    y = [res['ccf_median'].max() for res in data]

    p = np.polyfit(x, y, 1)
    return x, y, p


def plot_max_e_corr(res):
    x, y, p = res
    fit_fn = np.poly1d(p)

    x_fit = np.linspace(0, 1, 10)
    y_fit = fit_fn(x_fit)
    fig, ax = plt.subplots(1)

    ax.plot(x, y, 'x', label='$max(ccf_{median})$')
    ax.plot(x_fit, y_fit, label=f"lin. reg. fit b={p[0]}")
    ax.set_xlabel('e')
    ax.set_ylabel('$max(ccf_{median})$')
    ax.legend(loc="upper right")
    fig.suptitle('Correlation between e and $max(ccf_{median})$')


if __name__ == '__main__':
    start_time = time.perf_counter()

    results = load_results(Path.cwd() / 'results' / 'paramVsTau' / '200_1000_0')

    plot_max_e_corr(fit_max_e_corr(results))
    plt.savefig(Path.cwd() / 'src' / 'analyzation' / 'paramVsTau' / 'max_e_corr.pdf')



    plot_heatmap([res for res in results if res['p']['noiseType']['type'] == NoiseType.WHITE], 'tau1', 'e',
                 'CCF Peak height using white noise')
    plt.savefig(Path.cwd() / 'src' / 'analyzation' / 'paramVsTau' / 'white_noise.pdf')
    plot_heatmap([res for res in results if res['p']['noiseType']['type'] == NoiseType.RED], 'tau1', 'e',
                 'CCF Peak height using red noise')
    plt.savefig(Path.cwd() / 'src' / 'analyzation' / 'paramVsTau' / 'red_noise.pdf')

    print(f"It took {time.perf_counter() - start_time}ms to plot everything")
    plt.show()
