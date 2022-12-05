import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np


def calc_fft(array_in):
    array_diff = array_in[1:] - array_in[:-1]
    yf = abs(scipy.fft.fft(array_diff))
    xf = np.arange(len(yf)) / len(yf) * 7
    return xf, yf


def plot_fft(xf, yf, title, fig_name=None, figsize=(10, 6),
             dpi=300, to_show=True):
    plt.figure(figsize=figsize)
    cutoff = len(yf) // 2
    plt.plot(xf[:cutoff], yf[:cutoff])
    plt.xlabel('1 / week')

    plt.title(title)
    plt.grid()
    plt.tight_layout()

    if fig_name:
        plt.savefig(fig_name, dpi=dpi)

    if to_show:
        plt.show()
