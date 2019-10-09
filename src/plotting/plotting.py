import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.rcParams['figure.autolayout'] = False

def plt_samples(t, results):
    res_zero = results[0]
    legend = ['ou', 'underlying noise (white)']
    plt_2_graphs_with_same_axis(t, [res_zero['ou1'][0], res_zero['noise1'][0]],
                                [res_zero['ou2'][0], res_zero['noise2'][0]],
                                title1='OU Sample (Default)',
                                title2='OU Sample (Delayed)',
                                xlabel='t',
                                legends=[['ou', 'underlying noise (white)'], []])
    res_zero = results[6]
    plt_2_graphs_with_same_axis(t,
                                [res_zero['ou1'][0], res_zero['noise1'][0]],
                                [res_zero['ou2'][0], res_zero['noise2'][0]],
                                title1='OU Sample (Default)',
                                title2='OU Sample (Delayed)',
                                xlabel='t',
                                legends=[['ou', 'underlying noise (red)'], []])


def plt_2_graphs_with_same_axis(t, y1s, y2s, xlabel='', ylabel='', legends=[[], []], title1='', title2=''):
    fig, axs = plt.subplots(2,1)

    for y1 in y1s:
        axs[0].plot(t, y1)
    for y2 in y2s:
        axs[1].plot(t, y2)

    if (len(legends[0]) > 0):
        axs[0].legend(legends[0])

    if (len(legends[1]) > 0):
        axs[1].legend(legends[1])

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

def plt_time_series(params, ts, ys, title, labels=[], xlabel='', ylabel=''):
    cols = 3
    rows = int(np.ceil(len(ts)/cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=[12, 12])

    for i, [t, y] in enumerate(zip(ts, ys)):
        r = int(np.floor(i / cols))
        c = int(i % cols)

        showLabels = len(labels) > 0
        for j, _ in enumerate(t):
            label = labels[j] if showLabels else ''
            axs[r][c].plot(t[j], y[j], label=label)

        axs[r][c].title.set_text(f"e: {params[i]['e']}, tau: {params[i]['tau1']}")
        if showLabels:
            axs[r][c].legend(loc="upper right")
        axs[r][c].set_xlabel(xlabel)
        axs[r][c].set_ylabel(ylabel)

    st = fig.suptitle(title, size=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()