import os
import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import matplotlib as mpl
import netCDF4 as ncd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch import nn, optim
import matplotlib.cm as cm
import copy as copy
import multiprocessing as mp
from scipy import stats
import time as time
import matplotlib.font_manager
import seaborn as sns
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar
import xarray as xr
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from datetime import datetime
import warnings

np.random.seed(10)  # Set random seed for reproducibility

def corio(lat):
    return  2*(2*np.pi/(24*60*60)) * np.sin(lat*(np.pi/180))



def train_model(data_load, condition):
    
    ind1 = np.where(np.abs(data_load['heat'][:])<601)[0]
    ind2 = np.where(np.abs(data_load['tx'][:])<1.2)[0]
    ind3 = np.where(np.abs(data_load['h'][:])>29)[0]
    ind4 = np.where(np.abs(data_load['h'][:])<301)[0]
    ind5 = np.intersect1d(ind1, ind2)
    ind6 = np.intersect1d(ind3,ind5)
    ind7 = np.intersect1d(ind4,ind6)

    mm1=0; mm2=16

    data_forc=np.zeros([len(ind7),3])
    data_forc[:,0]=data_load['lat'][:][ind7]
    data_forc[:,1]=data_load['heat'][:][ind7]
    data_forc[:,2]=data_load['tx'][:][ind7]

    if condition == 'paper':                       
        data_load_main=np.zeros([len(data_load['h'][:][ind7]),4+mm2-mm1])
        data_load_main[:,0]=corio(data_load['l'][:])[ind7]
        data_load_main[:,1]=data_load['b0'][:][ind7]
        data_load_main[:,2]=data_load['ustar'][:][ind7]
        data_load_main[:,3]=data_load['h'][:][ind7]
        data_load_main[:,4:(mm2-mm1+4)]=data_load['SF'][:][ind7,mm1:mm2]
        
        data_load3=copy.deepcopy(data_load_main)

        print('started paper model')

        data, x,y, stats, k_mean, k_std= preprocess_train_data(data_load3) 

 #       x=torch.FloatTensor(x).to(device)
 #       y=torch.FloatTensor(y).to(device)
        
    elif condition == 'lat':
        data_load_main=np.zeros([len(data_load['h'][:][ind7]),5+mm2-mm1])
        data_load_main[:,0]=corio(data_load['l'][:])[ind7]
        data_load_main[:,1]=data_load['b0'][:][ind7]
        data_load_main[:,2]=data_load['ustar'][:][ind7]
        data_load_main[:,3]=data_load['h'][:][ind7]
        data_load_main[:,4]=data_load['lat'][:][ind7]
        data_load_main[:,5:(mm2-mm1+5)]=data_load['SF'][:][ind7,mm1:mm2]
    
        data_load3=copy.deepcopy(data_load_main)

        print('started model with latitude')

        data, x,y, stats, k_mean, k_std=preprocess_train_data_nd(data_load3) 


    elif condition == 'heat':
        data_load_main=np.zeros([len(data_load['h'][:][ind7]),5+mm2-mm1])
        data_load_main[:,0]=corio(data_load['l'][:])[ind7]
        data_load_main[:,1]=data_load['b0'][:][ind7]
        data_load_main[:,2]=data_load['ustar'][:][ind7]
        data_load_main[:,3]=data_load['h'][:][ind7]
        data_load_main[:,4]=data_load['heat'][:][ind7]
        data_load_main[:,5:(mm2-mm1+5)]=data_load['SF'][:][ind7,mm1:mm2]
    
        data_load3=copy.deepcopy(data_load_main)

        print('started model with heat')

        data, x,y, stats, k_mean, k_std=preprocess_train_data_nd(data_load3) 

    elif condition == 'wind':
        data_load_main=np.zeros([len(data_load['h'][:][ind7]),5+mm2-mm1])
        data_load_main[:,0]=corio(data_load['l'][:])[ind7]
        data_load_main[:,1]=data_load['b0'][:][ind7]
        data_load_main[:,2]=data_load['ustar'][:][ind7]
        data_load_main[:,3]=data_load['h'][:][ind7]
        data_load_main[:,4]=data_load['tx'][:][ind7]
        data_load_main[:,5:(mm2-mm1+5)]=data_load['SF'][:][ind7,mm1:mm2]
    
        data_load3=copy.deepcopy(data_load_main)

        print('started model with wind')

        data, x,y, stats, k_mean, k_std=preprocess_train_data_nd(data_load3) 

    elif condition == 'all':
        data_load_main=np.zeros([len(data_load['h'][:][ind7]),7+mm2-mm1])
        data_load_main[:,0]=corio(data_load['l'][:])[ind7]
        data_load_main[:,1]=data_load['b0'][:][ind7]
        data_load_main[:,2]=data_load['ustar'][:][ind7]
        data_load_main[:,3]=data_load['h'][:][ind7]
        data_load_main[:,4]=data_load['lat'][:][ind7]
        data_load_main[:,5]=data_load['heat'][:][ind7]
        data_load_main[:,6]=data_load['tx'][:][ind7]
        data_load_main[:,7:(mm2-mm1+7)]=data_load['SF'][:][ind7,mm1:mm2]
    
        data_load3=copy.deepcopy(data_load_main)

        print('started all data')

        data, x,y, stats, k_mean, k_std=preprocess_train_data_ad(data_load3) 
    
    return data, x, y, stats, k_mean, k_std

def preprocess_train_data_nd(data_load):
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

    new_mean, new_std = np.mean(data_load[:, 4]), np.std(data_load[:, 4])
    data_load[:, 4] = (data_load[:, 4] - new_mean) / new_std  # Standardize column 4
    # Log-transform and standardize the remaining columns (outputs)
    for j in range(len(data_load[:, 0])):
        data_load[j, 5:] = np.log(data_load[j, 5:] / np.max(data_load[j, 4:]))

    k_mean = np.mean(data_load[:, 5:], axis=0)
    k_std = np.std(data_load[:, 5:], axis=0)
    for k in range(data_load.shape[1] - 5):
        data_load[:, k + 5] = (data_load[:, k + 5] - k_mean[k]) / k_std[k]

    # Split into inputs (x) and outputs (y)
    x = data_load[ind_shuffle, :5]  # First 4 columns as input features
    y = data_load[ind_shuffle, 5:]  # Remaining columns as output labels

    # Return preprocessed data, statistics, and shuffle order
    stats = np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std, new_mean, new_std])
    return data_load, x, y, stats, k_mean, k_std


def preprocess_train_data_ad(data_load):
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

    new_mean, new_std = np.mean(data_load[:, 4]), np.std(data_load[:, 4])
    data_load[:, 4] = (data_load[:, 4] - new_mean) / new_std  # Standardize column 4
    # Log-transform and standardize the remaining columns (outputs)

    new_mean2, new_std2 = np.mean(data_load[:, 5]), np.std(data_load[:, 5])
    data_load[:, 5] = (data_load[:, 5] - new_mean2) / new_std2  # Standardize column 5
    # Log-transform and standardize the remaining columns (outputs)

    new_mean3, new_std3 = np.mean(data_load[:, 6]), np.std(data_load[:, 6])
    data_load[:, 6] = (data_load[:, 6] - new_mean3) / new_std3  # Standardize column 5
    # Log-transform and standardize the remaining columns (outputs)
    # Log-transform and standardize the remaining columns (outputs)
    for j in range(len(data_load[:, 0])):
        data_load[j, 7:] = np.log(data_load[j, 7:] / np.max(data_load[j, 7:]))

    k_mean = np.mean(data_load[:, 7:], axis=0)
    k_std = np.std(data_load[:, 7:], axis=0)
    for k in range(data_load.shape[1] - 7):
        data_load[:, k + 7] = (data_load[:, k + 7] - k_mean[k]) / k_std[k]

    # Split into inputs (x) and outputs (y)
    x = data_load[ind_shuffle, :7]  # First 4 columns as input features
    y = data_load[ind_shuffle, 7:]  # Remaining columns as output labels

    # Return preprocessed data, statistics, and shuffle order
    stats = np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std, new_mean, new_std, new_mean2, new_std2, new_mean3, new_std3])
    return data_load, x, y, stats, k_mean, k_std

def modeltrain_loss(In_nodes, Hid, Out_nodes, lr, epochs, x, y, valid_x, valid_y, model, k_std_y, k_mean, k_std, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr)  # Adam optimizer
    loss_fn = torch.nn.L1Loss(reduction='mean')  # L1 loss for gradient computation
    loss_array = torch.zeros([epochs, 3])  # Array to store epoch, train, and validation losses

    best_loss = float('inf')  # Initialize the best validation loss as infinity
    no_improvement = 0  # Counter for epochs without improvement
    best_model_state = None  # Placeholder for the best model state

    # Add a progress bar
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        for k in range(epochs):
            optimizer.zero_grad()  # Clear gradients from the previous step
            y_pred = model(x)  # Forward pass for training data
            
            valid_pred = model(valid_x)  # Forward pass for validation data
            
            # Loss used for gradient calculation
            loss = loss_fn(y_pred * k_std_y, y * k_std_y)
            
            loss_train = torch.mean(torch.abs(torch.exp(y_pred * k_std + k_mean) - torch.exp(y * k_std + k_mean)))
            loss_valid = torch.mean(torch.abs(torch.exp(valid_pred * k_std + k_mean) - torch.exp(valid_y * k_std + k_mean)))
            
            loss.backward()  # Backpropagate the gradient
            optimizer.step()  # Update model parameters

            # Record the losses for this epoch
            loss_array[k, 0] = k  
            loss_array[k, 1] = loss_train.item()  
            loss_array[k, 2] = loss_valid.item()  

            # Update the progress bar with the current epoch and losses
            pbar.set_postfix(
                train_loss=loss_train.item(), 
                valid_loss=loss_valid.item(), 
                patience_count=no_improvement
            )
            pbar.update(1)  # Increment the progress bar

            # Early stopping: Check if validation loss improves
            if loss_valid.item() < best_loss:
                best_loss = loss_valid.item()  # Update best loss
                no_improvement = 0
                best_model_state = model.state_dict()  
            else:
                no_improvement += 1  # Increment no improvement counter

            # If no improvement for 'patience' epochs, stop training
            if no_improvement >= patience:
                print(f"\nEarly stopping at epoch {k+1}. Validation loss has not improved for {patience} epochs.")
                break

            # Free memory by deleting intermediate variables
            del loss, y_pred
            
    # Restore the best model state after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, loss_array[:k, :]

def return_model_error(model, x, valid_x, y, valid_y, k_mean, k_std):
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

    return yerr, kappa_mean, ytestcpu, yptestcpu 

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

class learnKappa_layers(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # First layer: Input to hidden
        self.linear2 = nn.Linear(Hid, Hid)       # Second layer: Hidden to hidden
        self.linear3 = nn.Linear(Hid, Out_nodes) # Third layer: Hidden to output
        self.dropout = nn.Dropout(0.25)          # Dropout for regularization

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)            # ReLU activation for layer 1
        h1 = self.dropout(h1)          # Apply dropout
        
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)            # ReLU activation for layer 2
        h3 = self.dropout(h3)          # Apply dropout

        y_pred = self.linear3(h3)      # Final output layer
        return y_pred
