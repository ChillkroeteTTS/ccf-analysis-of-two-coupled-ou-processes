import matplotlib.pyplot as plt

def plt_noise(t, noise1, noise2):
    fig, axs = plt.subplots(2,1)
    fig.suptitle('generated noise')
    axs[0].plot(t, noise1)
    axs[1].plot(t, noise2)
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='noise')
    plt.show()

def plt_ou(t, ou):
    fig = plt.figure()
    plt.suptitle('Ohrnstein Uhlenbeck processs')

    plt.plot(t, ou)
    plt.show()