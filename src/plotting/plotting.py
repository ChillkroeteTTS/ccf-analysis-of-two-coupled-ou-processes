import matplotlib.pyplot as plt

def plt_noise(t, noise1, noise2):
    fig, axs = plt.subplots(2,1)
    fig.suptitle('generated noise')
    plt_2_graphs_with_same_axis(fig, axs, t, noise1, noise2)

def plt_ou(t, ou1, ou2):
    fig, axs = plt.subplots(2,1)
    fig.suptitle('Ohrnstein Uhlenbeck processes')
    plt_2_graphs_with_same_axis(fig, axs, t, ou1, ou2, legends=[['ou1'], ['ou2 (mixed)']])

def plt_2_graphs_with_same_axis(fig, axs, t, y1, y2, xlabel='', ylabel='', legends=[[], []]):
    axs[0].plot(t, y1)
    axs[1].plot(t, y2)
    axs[0].legend(legends[0])
    axs[1].legend(legends[1])
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='noise')
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