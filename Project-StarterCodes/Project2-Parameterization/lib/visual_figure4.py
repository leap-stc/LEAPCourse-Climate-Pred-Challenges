import os
import numpy as np
import matplotlib.pyplot as plt
import copy

def corio(lat):
    """Calculate the Coriolis parameter for a given latitude."""
    return 2 * (2 * np.pi / (24 * 60 * 60)) * np.sin(lat * (np.pi / 180))

def get_hist(y, k_mean, k_std):
    """Get histogram values for normalized data."""
    vals, binss = np.histogram(np.exp(y * k_std + k_mean), range=(0, 1.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])

def get_hist2(y):
    """Get histogram values for error data."""
    vals, binss = np.histogram(y, range=(-0.2, 0.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])

def performance_sigma_point(model, x, valid_x, y, valid_y, k_mean, k_std):
    """Plot the performance of a neural network model.

    Parameters:
        model: Trained neural network model.
        x: Training input data.
        valid_x: Validation input data.
        y: Training output data.
        valid_y: Validation output data.
        k_mean: Mean normalization values.
        k_std: Standard deviation normalization values.
    """
    # plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'  # ensures it can math compatibility with symbols in your code without erroring fix no cursive_fontsystem


    y_pred_train = model(x)
    y_pred_test = model(valid_x)

    ycpu = y.cpu().detach().numpy()
    ytestcpu = valid_y.cpu().detach().numpy()
    yptraincpu = y_pred_train.cpu().detach().numpy()
    yptestcpu = y_pred_test.cpu().detach().numpy()

    ystd = np.zeros(16)
    yteststd = np.zeros(16)
    ypstd = np.zeros(16)
    ypteststd = np.zeros(16)
    yerr = np.zeros(16)
    kappa_mean = np.zeros(16)

    for i in range(16):
        ystd[i] = np.std(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))
        yteststd[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]))
        ypstd[i] = np.std(np.exp(yptraincpu[:, i] * k_std[i] + k_mean[i]))
        ypteststd[i] = np.std(np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))
        yerr[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))

        kappa_mean[i] = np.mean(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))

    plt.figure(figsize=(15, 10))

    ind = np.arange(0, 16)
    ind_tick = np.arange(1, 17)[::-1]

    # Subplot 1: Boxplot of network output differences
    plt.subplot(1, 4, 1)
    for i in range(16):
        plt.boxplot(ytestcpu[:, i] - yptestcpu[:, i], vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.xlim([-2.0, 2.0])
    plt.yticks(ind, ind_tick)
    plt.title(r'(a) Output of network $\mathcal{N}_1$ ')
    plt.ylabel('Node')

    # Subplot 2: Boxplot of shape function differences
    plt.subplot(1, 4, 2)
    for i in range(16):
        plt.boxplot(kappa_mean[i] + np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]),
                    vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.yticks([])
    plt.title(r'(b) Shape function $g(\sigma)$')
    plt.xlabel(r'$g(\sigma)$')

    # Subplots 3 & 4: Histograms
    k12 = 15
    for k in range(16):
        plt.subplot(16, 4, 4 * k + 3)
        vals, binss = get_hist(ytestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color='blue')

        vals, binss = get_hist(yptestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color='red')
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(c) Probability density histogram')

        plt.subplot(16, 4, 4 * k + 4)
        vals, binss = get_hist2(np.exp(ytestcpu[:, k12] * k_std[k12] + k_mean[k12]) - np.exp(yptestcpu[:, k12] * k_std[k12] + k_mean[k12]))
        plt.plot(binss, vals, color='green')
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(d) Error histogram ')

        k12 -= 1

    plt.tight_layout()
    # print("Plot saved as 'modelstats.pdf'")
