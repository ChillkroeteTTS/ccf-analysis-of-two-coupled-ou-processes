import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mlp
from pandas import DataFrame
from scipy.stats import norm
from typing import List
from noise import NoiseType
from stats import PercentileResult, SimulationResults

mlp.rcParams['figure.autolayout'] = False


def plt_sample_from_ensemble(t, t1, p, ensemble: List[DataFrame]):
    default_noise_intensity = np.median([df['noise1'].var() for df in ensemble])
    mixed_noise_intensity = np.median([df['mixed_noise'].var() for df in ensemble])
    noise_type = p['noiseType']['type']
    ensemble[0][['ou1', 'ou2']] \
        .rename({'ou1': f'OU1: {noise_type} (var: {default_noise_intensity:.2f})',
                 'ou2': f'OU2: {noise_type} (var: {mixed_noise_intensity:.2f})'}, axis='columns') \
        .plot(title=f"Sample from ensemble run with e={p['e']}, $\\tau_1$={p['tau1']}, $\\tau_2$={p['tau2']}") \
        .set_xlabel('t')


def plt_2_graphs_with_same_axis(t, y1s, y2s, xlabel='', ylabel='', legends=[[], []], title1='', title2='',
                                percentiles1=None, percentiles2=None):
    fig, axs = plt.subplots(2, 1, figsize=[12, 12])

    if percentiles1 is not None:
        curves = np.squeeze(percentiles1)
        axs[0].fill_between(t, curves[0], curves[1], alpha=0.4)
    for y1 in y1s:
        axs[0].plot(t, y1)

    if percentiles2 is not None:
        curves = np.squeeze(percentiles2)
        axs[1].fill_between(t, curves[0], curves[1], alpha=0.4)
    for y2 in y2s:
        axs[1].plot(t, y2)

    if len(legends[0]) > 0:
        axs[0].legend(legends[0], loc='upper right')

    if len(legends[1]) > 0:
        axs[1].legend(legends[1], loc='upper right')

    axs[0].title.set_text(title1)
    axs[1].title.set_text(title2)

    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()


def plt_acf(y1, y2):
    fig, axs = plt.subplots(2, 1, sharey=True)
    fig.suptitle('Auto Correlation Functions')
    axs[0].plot(y1)
    axs[1].plot(y2)
    axs[0].legend(['ACF of ou1'])
    axs[1].legend(['ACF of ou2'])
    for ax in axs.flat:
        ax.axhline(0, linestyle='--', color='red')
    # plt.tight_layout()
    plt.show()


def standart_title(params, i):
    return f"e: {params[i]['e']}, tau: {params[i]['tau1']}, gamma: {params[i].get('noiseType').get('gamma1')}"


def plot_heatmap(results: List[SimulationResults], xkey, ykey, title, fig=None, ax=None):
    z = gen_3d_data(results, xkey, ykey)

    if ax is None or fig is None:
        fig, ax = plt.subplots(1)
    plt.pcolormesh(np.arange(0, 1.2, 0.1), np.arange(0, 1.2, 0.1), z, vmin=0, vmax=1, cmap='OrRd')
    plt.colorbar()
    fig.suptitle(title)
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    return z


def gen_3d_data(results, xkey, ykey):
    z = np.zeros([11, 11])  # 0, 0.1, ..., 1 (inclusive)
    for res in results:
        ix = int(res['p'][xkey] * 10)
        iy = int(res['p'][ykey] * 10)
        z[iy][ix] = res['ccf_median'].max()
    return z


def gen_2d_data(results, xkey):
    z = np.zeros(11)  # 0, 0.1, ..., 1 (inclusive)
    for res in results:
        ix = int(res['p'][xkey] * 10)
        z[ix] = res['ccf_median'].max()
    return z


def plot_with_percentiles(results: DataFrame, ax, label, xlabel='', ylabel='', prefix='', suffix=''):
    median = results[prefix+'median'+suffix]
    ax.fill_between(median.index, results[prefix+'lower_percentile'+suffix], results[prefix+'upper_percentile'+suffix], alpha=0.4)
    ax.plot(median.index, median, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")


def plot_multiple_percentiles(results: List[List[PercentileResult]], xlabel, ylabel, labels: List[str], title='', prefix='', suffix=''):
    cols = min(len(results), 3)
    rows = int(np.ceil(len(results) / cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, squeeze=False)
    for i, res in enumerate(results):
        r = int(np.floor(i / cols))
        c = int(i % cols)
        plot_with_percentiles(res, axs[r][c], labels, xlabel, ylabel)

    fig.suptitle(title)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.01, hspace=0.01)
    plt.subplots_adjust(hspace=1)


# deprecated
def plt_time_series(params, ts, ys, title, subTitleFn=standart_title, labels=[], xlabel='', ylabel='',
                    percentiles_per_run=[]):
    cols = min(len(ts), 3)
    rows = int(np.ceil(len(ts) / cols))

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=[4 * cols, 4 * rows], squeeze=False)
    for i, [t, y, percentiles] in enumerate(
            zip(ts, ys, [percentiles_per_run[i] if len(percentiles_per_run) > i else None for i, _ in enumerate(ts)])):
        r = int(np.floor(i / cols))
        c = int(i % cols)

        for j, _ in enumerate(t):
            showLabels = (len(labels) > 0 and isinstance(labels[0], str)) or (len(labels) > i and len(labels[i]) > j)
            label = labels[j] if showLabels and isinstance(labels[0], str) else (
                labels[i][j] if showLabels and isinstance(labels[0], list) else '')
            if percentiles is not None and len(percentiles) > j:
                curves = np.squeeze(percentiles[j])
                axs[r][c].fill_between(t[j], curves[0], curves[1], alpha=0.4)
            axs[r][c].plot(t[j], y[j], label=label)

        axs[r][c].title.set_text(subTitleFn(params, i))

        if showLabels:
            axs[r][c].legend(loc="upper right")

        axs[r][c].set_xlabel(xlabel)
        axs[r][c].set_ylabel(ylabel)

        if i == (cols - 1):
            st = fig.suptitle(title, size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.show()


def plt_correlation(results):
    def fit_norm(r):
        ccf_median = r['ccf_ensemble']['ccf_median']
        mu, std = norm.fit(np.squeeze(ccf_median))
        # move fitted normal distribution from center
        x_middle = ccf_median.index[int(ccf_median.index.size / 2)]
        mu = mu + x_middle
        x = np.linspace(ccf_median.iloc[0], ccf_median.iloc[-1], 500)
        p = norm.pdf(x, mu, std)
        return [[x, p], [mu, std]]

    def get_param_corr_pairs(getParam, getFixedParam1, getFixedParam2, yFn):
        def isNoiseType(r, type):
            return r['p']['noiseType']['type'] == type

        def same_fixed_param_and_noise_type(r: SimulationResults, noiseType):
            red_noise_params = [r for r in results if isNoiseType(r, NoiseType.RED)]
            firstValueFixedParam1 = getFixedParam1(red_noise_params[0]['p'])
            firstValueFixedParam2 = getFixedParam2(red_noise_params[0]['p'])
            return isNoiseType(r, noiseType) \
                   and (getFixedParam1(r['p']) == None or getFixedParam1(r['p']) == firstValueFixedParam1) \
                   and (getFixedParam2(r['p']) == None or getFixedParam2(r['p']) == firstValueFixedParam2)

        parameters_white = [getParam(r['p']) for r in results if same_fixed_param_and_noise_type(r, NoiseType.WHITE)]
        parameters_red = [getParam(r['p']) for r in results if same_fixed_param_and_noise_type(r, NoiseType.RED)]
        print(parameters_red)
        max_ccf_values_white = [yFn(r) for r in results if same_fixed_param_and_noise_type(r, NoiseType.WHITE)]
        max_ccf_values_red = [yFn(r) for r in results if same_fixed_param_and_noise_type(r, NoiseType.RED)]

        return [[parameters_white, max_ccf_values_white], [parameters_red, max_ccf_values_red]]

    getE = lambda p: p['e']
    getTau1 = lambda p: p['tau1']
    getGamma = lambda p: p['noiseType'].get('gamma1')

    [[es_white, maxs_ccf_white], [es_red, maxs_ccf_red]] = get_param_corr_pairs(getE, getTau1, getGamma,
                                                                                lambda r: np.max(r['ccf_ensemble']['ccf_median']))
    [[gammas_white, std_norm_white], [gammas_red, std_norm_red]] = get_param_corr_pairs(getGamma, getTau1, getE,
                                                                                        lambda r: fit_norm(r)[1][1])

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(es_white, maxs_ccf_white, 'x-')
    axs[0].plot(es_red, maxs_ccf_red, 'x-', color='r')

    axs[0].title.set_text(f"Correlation between max(ccf) and e")
    axs[0].set_xlabel(f"e")
    axs[0].set_ylabel(f"max(ccf)")
    axs[0].legend([f"white noise", f"red noise"])

    axs[1].plot(gammas_white, std_norm_white, 'x')
    axs[1].plot(gammas_red, std_norm_red, 'x', color='r')

    axs[1].title.set_text(f"Correlation between std of a fitted normal distribution and gamma")
    axs[1].set_xlabel(f"gamma")
    axs[1].set_ylabel(f"std(normfit(ccf))")
    axs[1].legend([f"white noise driven", f"red noise driven"])

    # plt.tight_layout()
    plt.show()


def fwahh(ts):
    hh = ts.max() / 2
    i_hh = ts.idxmax()

    i_l = None
    i_r = None
    x_l = None
    x_r = None

    for x in range(0, ts.index.size):
        i = ts.index[x]
        if i_l == None and ts[i] > hh:
            i_l = i
            x_l = x
        if i_r == None and i_l is not None and ts[i] < hh:
            i_r = i
            x_r = x

        if i_hh > i and i_l is not None and i_r is not None and ts[i] > hh:
            i_l = i
            x_l = x
            i_r = None
            x_r = None

    return [i_l, i_r], [x_l, x_r], hh

def plot_fwahh(ax, ts):
    [i_l, i_r], [x_l, x_r], hh = fwahh(ts)

    h_l = ts[i_l]
    h_r = ts[i_r]

    i_l_i = (ts.index[max(0, x_l + (-1 if h_l > h_r else 1))] + ts.index[x_l]) / 2
    i_r_i = (ts.index[x_r + (-1 if h_l > h_r else 1)] + ts.index[x_r]) / 2
    h_l_i = (ts.iloc[max(0, x_l + (-1 if h_l > h_r else 1))] + ts.iloc[x_l]) / 2
    h_r_i = (ts.iloc[x_r + (-1 if h_l > h_r else 1)] + ts.iloc[x_r]) / 2

    # ax.plot(x_peak_index, height, 'ro')
    ax.plot([i_l_i, i_r_i], [(h_l_i + h_r_i) / 2, (h_l_i + h_r_i) / 2], 'r--')
    # ax.plot(i_l_i, h_l_i, 'yo')

    # x_debug = x_r + (-1 if h_l > h_r else 1)
    # ax.plot(ts.index[x_debug], ts.iloc[x_debug], 'yo')
    # ax.plot(500, hh, 'yo')

    # ax.plot(i_l_i, h_l_i, 'go')
    # ax.plot(i_r_i, h_r_i, 'go')
    # ax.plot(i_l, ts[i_l], 'ro')
    # ax.plot(i_r, ts[i_r], 'ro')
