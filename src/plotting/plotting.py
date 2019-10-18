import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mlp

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
    plt.tight_layout()
    plt.show()

def plt_time_series(params, ts, ys, title, labels=[], xlabel='', ylabel='', percentiles_per_run=[]):
    cols = 3
    rows = int(np.ceil(len(ts)/cols))
    plt.subplots_adjust(top=2)
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=[12, 12])

    for i, [t, y, percentiles] in enumerate(zip(ts, ys, [percentiles_per_run[i] if len(percentiles_per_run) > i else None for i, _ in enumerate(ts)])):
        r = int(np.floor(i / cols))
        c = int(i % cols)

        showLabels = len(labels) > 0
        for j, _ in enumerate(t):
            label = labels[j] if showLabels else ''
            if percentiles is not None and len(percentiles) > j:
                curves = np.squeeze(percentiles[j])
                axs[r][c].fill_between(t[j], curves[0], curves[1], alpha=0.4)
            axs[r][c].plot(t[j], y[j], label=label)

        axs[r][c].title.set_text(f"e: {params[i]['e']}, tau: {params[i]['tau1']}")

        if showLabels:
            axs[r][c].legend(loc="upper right")

        axs[r][c].set_xlabel(xlabel)
        axs[r][c].set_ylabel(ylabel)

    plt.subplots_adjust(top=2)
    st = fig.suptitle(title, size=16)
    plt.subplots_adjust(top=2)
    plt.tight_layout()
    plt.subplots_adjust(top=2)
    plt.show()