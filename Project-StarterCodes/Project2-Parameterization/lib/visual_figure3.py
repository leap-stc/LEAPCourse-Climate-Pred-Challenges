import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

import matplotlib.pyplot as plt

today = datetime.today()
np.random.seed(100)

def score_eval(hidds, lays, valid_x, valid_y, k_mean_c, k_std_c, cwd_output):
    """
    Evaluate model performance by calculating scores for different architectures
    and saving training and validation losses.

    Parameters:
        hidds (list): List of hidden layer sizes.
        lays (list): List of layer counts.
        valid_x (torch.Tensor): Validation input data.
        valid_y (torch.Tensor): Validation target data.
        k_mean_c (torch.Tensor): Mean values for normalization.
        k_std_c (torch.Tensor): Standard deviation values for normalization.
        cwd_output (str): Output directory to save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = np.zeros([len(hidds), len(lays)])
    p = torch.zeros(16)

    # Evaluate model scores
    for k, h in enumerate(hidds):
        for l, la in enumerate(lays):
            model_path = os.path.join(cwd_output, f'ensemble_models_layers{la}/mod_dir_{h}/model.pt')
            model = torch.load(model_path, map_location=device)
            model = model.to(device)
            y = model(valid_x)

            for i in range(16):
                asd = (torch.exp(y[:, i] * k_std_c[i] + k_mean_c[i]) - valid_y[:, i]).detach().cpu().numpy()
                asd1 = np.percentile(asd, 5)
                asd2 = np.percentile(asd, 95)
                ind_iqr = np.intersect1d(np.where(asd > asd1)[0], np.where(asd < asd2)[0])

                y_new = torch.exp(y[ind_iqr, i] * k_std_c[i] + k_mean_c[i])
                p[i] = torch.corrcoef(torch.stack((y_new, valid_y[ind_iqr, i]), 0))[0, 1]

            score[k, l] = torch.mean(p).item()

    # Save results
    os.makedirs(os.path.join(cwd_output, 'n1scoredata'), exist_ok=True)
    np.savetxt(os.path.join(cwd_output, 'n1scoredata/N1scores.txt'), score)

    print("Scores saved successfully.")

def save_losses_by_seed(cwd_sd, seeds):
    """
    Save training and validation losses separately for each seed.

    Parameters:
        cwd_sd (str): Directory containing loss data for each seed.
        seeds (list): List of seed values.
    """
    os.makedirs(os.path.join(cwd_sd, 'loss_by_seed'), exist_ok=True)
    for k in seeds:
        loss_path = os.path.join(cwd_sd, f'mod_dir_{k}/loss_array.txt')
        losses = np.loadtxt(loss_path)
        np.savetxt(os.path.join(cwd_sd, f'loss_by_seed/loss_seed_{k}.txt'), losses)
    print("Losses saved by seed successfully.")

def load_losses_by_seed(cwd_sd, seeds):
    """
    Load training and validation losses for all seeds.

    Parameters:
        cwd_sd (str): Directory containing loss data by seed.
        seeds (list): List of seed values.

    Returns:
        tuple: Two lists containing training and validation losses for all seeds.
    """
    tr_ls_list = []
    va_ls_list = []

    for k in seeds:
        loss_path = os.path.join(cwd_sd, f'loss_by_seed/loss_seed_{k}.txt')
        losses = np.loadtxt(loss_path)
        tr_ls_list.append(losses[:, 1])
        va_ls_list.append(losses[:, 2])

    return tr_ls_list, va_ls_list

def plot_n1_scores(cwd_output, hidds, lays, seeds, epochs=3000):
    """
    Function to plot linear correlation coefficients and loss metrics from given data files.

    Parameters:
        cwd_output (str): Output directory containing the results.
        hidds (list): List of hidden layer sizes.
        lays (list): List of layer counts.
        seeds (list): List of seed values.
        epochs (int): Maximum number of epochs for plotting loss metrics.
    """
    # Load score data
    score = np.loadtxt(os.path.join(cwd_output, 'n1scoredata/N1scores.txt'))

    # Load losses
    cwd_sd = os.path.join(cwd_output, 'ensemble_models_layers2_uncertainty/')
    tr_ls_list, va_ls_list = load_losses_by_seed(cwd_sd, seeds)

    # Handle NaN padding for variable-length losses
    tr_ls = np.array([np.pad(tr, (0, epochs - len(tr)), constant_values=np.nan) for tr in tr_ls_list]).T
    va_ls = np.array([np.pad(va, (0, epochs - len(va)), constant_values=np.nan) for va in va_ls_list]).T

    epchs = np.arange(0, epochs)

    # Filter valid rows
    valid_rows_tr = ~np.isnan(tr_ls).all(axis=1)
    valid_rows_va = ~np.isnan(va_ls).all(axis=1)

    # Configure plot appearance
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(16, 8))

    # Plot Linear Correlation Coefficients
    plt.subplot(121)
    plt.plot(hidds, score[:, 0], 'o', label='1 layer')
    plt.plot(hidds, score[:, 1], 'o', label='2 layers')

    hidds_labels = hidds
    plt.xticks(hidds, hidds_labels)
    plt.ylim([0, 1])
    plt.legend(fontsize=15, loc=4)
    plt.xlabel('Nodes in each hidden layer')
    plt.ylabel(r'Linear Correlation Coefficient')  # Pearson Correlation
    plt.annotate('(a)', (0.1, 0.9), xycoords='axes fraction')

    # Plot Loss Metrics
    plt.subplot(122)
    plt.fill_between(
        epchs[valid_rows_tr],
        np.nanmin(tr_ls[valid_rows_tr], axis=1),
        np.nanmax(tr_ls[valid_rows_tr], axis=1),
        alpha=0.1, color='m'
    )
    plt.fill_between(
        epchs[valid_rows_va],
        np.nanmin(va_ls[valid_rows_va], axis=1),
        np.nanmax(va_ls[valid_rows_va], axis=1),
        alpha=0.1, color='g'
    )
    plt.fill_between(
        epchs[valid_rows_va & valid_rows_tr],
        np.nanmin(va_ls[valid_rows_va] - tr_ls[valid_rows_tr], axis=1),
        np.nanmax(va_ls[valid_rows_va] - tr_ls[valid_rows_tr], axis=1),
        alpha=0.1, color='b'
    )

    plt.plot(epchs[valid_rows_tr], np.nanmean(tr_ls[valid_rows_tr], axis=1), 'm-', label='Training Loss')
    plt.plot(epchs[valid_rows_va], np.nanmean(va_ls[valid_rows_va], axis=1), 'g-', label='Validation Loss')
    plt.plot(epchs[valid_rows_va & valid_rows_tr], np.nanmean(va_ls[valid_rows_va] - tr_ls[valid_rows_tr], axis=1), 'b-', label='(Validation-Training) Loss')

    plt.legend(fontsize=15)
    plt.xlabel('Epochs')
    plt.ylabel(r'L1 loss')
    plt.annotate('(b)', (0.1, 0.9), xycoords='axes fraction')

    # Save the plot
    # plt.savefig(os.path.join(cwd_output, 'n1scoredata/N1sweepscore.pdf'), format='pdf')
    plt.show()
    plt.close()
    # print("Plot saved successfully.")
