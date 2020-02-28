import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.stats import norm

from noise import NoiseType

mlp.rcParams['figure.autolayout'] = False

def plt_samples(t,t1, results):
    res_zero = results[0]
    default_noise_intensity = np.median([np.var(n[t1:]) for n in res_zero['noise1']])
    mixed_noise_intensity = np.median([np.var(n[t1:]) for n in res_zero['noise2']])

    plt_2_graphs_with_same_axis(t, [res_zero['ou1'][0], res_zero['noise1'][0]],
                                [res_zero['ou2'][0], res_zero['noise2'][0]],
                                title1='OU Sample (Default)',
                                title2='OU Sample (Delayed)',
                                xlabel='t',
                                legends=[['ou', f'white noise (intensity: {default_noise_intensity:.2f})'],
                                         ['ou', f'white noise (intensity: {mixed_noise_intensity:.2f})']],
                                )
    res_zero = results[6]
    plt_2_graphs_with_same_axis(t,
                                [res_zero['ou1'][0], res_zero['noise1'][0]],
                                [res_zero['ou2'][0], res_zero['noise2'][0]],
                                title1='OU Sample (Default)',
                                title2='OU Sample (Delayed)',
                                xlabel='t',
                                legends=[['ou', f'red noise (intensity: {default_noise_intensity:.2f})'],
                                         ['ou', f'red noise (intensity: {mixed_noise_intensity:.2f})']])


def plt_2_graphs_with_same_axis(t, y1s, y2s, xlabel='', ylabel='', legends=[[], []], title1='', title2='', percentiles1=None, percentiles2=None):
    fig, axs = plt.subplots(2,1, figsize=[12,12])

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

    if (len(legends[0]) > 0):
        axs[0].legend(legends[0], loc='upper right')

    if (len(legends[1]) > 0):
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

def plt_time_series(params, ts, ys, title, subTitleFn=standart_title, labels=[], xlabel='', ylabel='', percentiles_per_run=[]):
    cols = min(len(ts), 3)
    rows = int(np.ceil(len(ts)/cols))

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=[4*cols, 4*rows], squeeze=False)
    for i, [t, y, percentiles] in enumerate(zip(ts, ys, [percentiles_per_run[i] if len(percentiles_per_run) > i else None for i, _ in enumerate(ts)])):
        r = int(np.floor(i / cols))
        c = int(i % cols)

        for j, _ in enumerate(t):
            showLabels = (len(labels) > 0 and isinstance(labels[0], str)) or (len(labels) > i and len(labels[i]) > j)
            label = labels[j] if showLabels and isinstance(labels[0], str) else (labels[i][j] if showLabels and isinstance(labels[0], list) else '')
            if percentiles is not None and len(percentiles) > j:
                curves = np.squeeze(percentiles[j])
                axs[r][c].fill_between(t[j], curves[0], curves[1], alpha=0.4)
            axs[r][c].plot(t[j], y[j], label=label)

        axs[r][c].title.set_text(subTitleFn(params, i))

        if showLabels:
            axs[r][c].legend(loc="upper right")

        axs[r][c].set_xlabel(xlabel)
        axs[r][c].set_ylabel(ylabel)

        if i == (cols-1):
            st = fig.suptitle(title, size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.show()

def plt_correlation(results):

    def fit_norm(r):
        mu, std = norm.fit(np.squeeze(r['ccf']))
        # move fitted normal distribution from center
        x_middle = r['ccf_shifts'][int(r['ccf_shifts'].size/2)]
        mu = mu + x_middle
        x = np.linspace(r['ccf_shifts'][0], r['ccf_shifts'][-1], 500)
        p = norm.pdf(x, mu, std)
        return [[x, p], [mu, std]]

    def get_param_corr_pairs(getParam, getFixedParam1, getFixedParam2, yFn):
        def isNoiseType(r, type):
            return r['params']['noiseType']['type'] == type

        def same_fixed_param_and_noise_type (r, noiseType):
            red_noise_params = [r for r in results if isNoiseType(r, NoiseType.RED)]
            firstValueFixedParam1 = getFixedParam1(red_noise_params[0]['params'])
            firstValueFixedParam2 = getFixedParam2(red_noise_params[0]['params'])
            return isNoiseType(r, noiseType) \
                   and (getFixedParam1(r['params']) == None or getFixedParam1(r['params']) == firstValueFixedParam1) \
                   and (getFixedParam2(r['params']) == None or getFixedParam2(r['params']) == firstValueFixedParam2)

        parameters_white = [getParam(r['params']) for r in results if same_fixed_param_and_noise_type(r, NoiseType.WHITE)]
        parameters_red = [getParam(r['params']) for r in results if same_fixed_param_and_noise_type(r, NoiseType.RED)]
        print(parameters_red)
        max_ccf_values_white = [yFn(r) for r in results if same_fixed_param_and_noise_type(r, NoiseType.WHITE)]
        max_ccf_values_red = [yFn(r) for r in results if same_fixed_param_and_noise_type(r, NoiseType.RED)]

        return [[parameters_white, max_ccf_values_white], [parameters_red, max_ccf_values_red]]

    getE = lambda params: params['e']
    getTau1 = lambda params: params['tau1']
    getGamma = lambda params: params['noiseType'].get('gamma1')

    [[es_white, maxs_ccf_white], [es_red, maxs_ccf_red]] = get_param_corr_pairs(getE, getTau1, getGamma, lambda r: np.max(r['ccf']))
    [[gammas_white, std_norm_white], [gammas_red, std_norm_red]] = get_param_corr_pairs(getGamma, getTau1, getE, lambda r: fit_norm(r)[1][1])

    fig, axs = plt.subplots(2,1)
    axs[0].plot(es_white, maxs_ccf_white, 'x')
    axs[0].plot(es_red, maxs_ccf_red, 'x', color='r')

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
