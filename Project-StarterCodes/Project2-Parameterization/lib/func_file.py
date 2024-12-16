import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import netCDF4 as ncd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.cm as cm

np.random.seed(10)  # Set random seed for reproducibility

# Define feedforward neural networks with different numbers of hidden layers.
# Each class corresponds to a model with an increasing number of hidden layers:
# - learnKappa_layers1: 1 hidden layer
# - learnKappa_layers2: 2 hidden layers
# - learnKappa_layers3: 3 hidden layers
# - learnKappa_layers4: 4 hidden layers
# Parameters:
# - In_nodes: Number of input nodes
# - Hid: Number of hidden layer nodes
# - Out_nodes: Number of output nodes

class learnKappa_layers1(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers1, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # Input to hidden layer
        self.linear2 = nn.Linear(Hid, Out_nodes)  # Hidden to output layer
        self.dropout = nn.Dropout(0.25)  # Dropout to reduce overfitting

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)  # ReLU activation
        h1 = self.dropout(h1)
        y_pred = self.linear2(h1)  # Output predictions
        return y_pred


class learnKappa_layers2(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers2, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        y_pred = self.linear3(h3)
        return y_pred


class learnKappa_layers3(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers3, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        y_pred = self.linear4(h5)
        return y_pred


class learnKappa_layers4(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers4, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Hid)
        self.linear5 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        h6 = self.linear4(h5)
        h7 = torch.relu(h6)
        h7 = self.dropout(h7)
        y_pred = self.linear5(h7)
        return y_pred


# Data preprocessing function
# Purpose: Standardizes input data, shuffles samples, and prepares features (x) and labels (y) for training.
def preprocess_train_data(data_load):
    # Create and shuffle indices
    ind = np.arange(0, len(data_load), 1)
    ind_shuffle = copy.deepcopy(ind)
    np.random.shuffle(ind_shuffle)

    # Standardize the first 4 columns (features)
    l_mean, l_std = np.mean(data_load[:, 0]), np.std(data_load[:, 0])
    data_load[:, 0] = (data_load[:, 0] - l_mean) / l_std  # Standardize column 0 (l)

    h_mean, h_std = np.mean(data_load[:, 1]), np.std(data_load[:, 1])
    data_load[:, 1] = (data_load[:, 1] - h_mean) / h_std  # Standardize column 1 (b0)

    t_mean, t_std = np.mean(data_load[:, 2]), np.std(data_load[:, 2])
    data_load[:, 2] = (data_load[:, 2] - t_mean) / t_std  # Standardize column 2 (u*)

    hb_mean, hb_std = np.mean(data_load[:, 3]), np.std(data_load[:, 3])
    data_load[:, 3] = (data_load[:, 3] - hb_mean) / hb_std  # Standardize column 3 (w*)

    # Log-transform and standardize the remaining columns (outputs)
    for j in range(len(data_load[:, 0])):
        data_load[j, 4:] = np.log(data_load[j, 4:] / np.max(data_load[j, 4:]))

    k_mean = np.mean(data_load[:, 4:], axis=0)
    k_std = np.std(data_load[:, 4:], axis=0)
    for k in range(data_load.shape[1] - 4):
        data_load[:, k + 4] = (data_load[:, k + 4] - k_mean[k]) / k_std[k]

    # Split into inputs (x) and outputs (y)
    x = data_load[ind_shuffle, :4]  # First 4 columns as input features
    y = data_load[ind_shuffle, 4:]  # Remaining columns as output labels

    # Return preprocessed data, statistics, and shuffle order
    stats = np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std])
    return data_load, x, y, stats, k_mean, k_std
