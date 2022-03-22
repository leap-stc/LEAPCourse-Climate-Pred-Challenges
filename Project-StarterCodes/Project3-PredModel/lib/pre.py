# libraries
import os
from pathlib import Path
from collections import defaultdict
import scipy
import random
import numpy as np
import xarray as xr
import pandas as pd
import joblib
from skimage.filters import sobel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout
from statsmodels.nonparametric.smoothers_lowess import lowess

#===============================================
#global path
#===============================================
root_dir = os.getcwd() # get the current wokring directory
#===============================================
# Masks
#===============================================

def network_mask():
    '''network_mask
    This masks out regions in the 
    NCEP land-sea mask (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
    to define the open ocean. Regions removed include:
    - Coast : defined by sobel filter
    - Batymetry less than 100m
    - Arctic ocean : defined as North of 79N
    - Hudson Bay
    - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
    '''
    ### Load obs directory
    dir_obs = f'{root_dir}/../data/artemis/observations'
    
    ### topography
    ds_topo = xr.open_dataset(f'{root_dir}/../other_file/GEBCO_2014_1x1_global.nc')
    ds_topo = ds_topo.roll(lon=180, roll_coords='lon')
    ds_topo['lon'] = np.arange(0.5, 360, 1)

    ### Loads grids
    # land-sea mask
    # land=0, sea=1
    ds_lsmask = xr.open_dataset(f'{root_dir}/../other_file/lsmask.nc').sortby('lat').squeeze().drop('time')
    data = ds_lsmask['mask'].where(ds_lsmask['mask']==1)
    ### Define Latitude and Longitude
    lon = ds_lsmask['lon']
    lat = ds_lsmask['lat']
    
    ### Remove coastal points, defined by sobel edge detection
    coast = (sobel(ds_lsmask['mask'])>0)
    data = data.where(coast==0)
    
    ### Remove shallow sea, less than 100m
    ### This picks out the Solomon islands and Somoa
    data = data.where(ds_topo['Height']<-100)
    
    ### remove arctic
    data = data.where(~((lat>79)))
    data = data.where(~((lat>67) & (lat<80) & (lon>20) & (lon<180)))
    data = data.where(~((lat>67) & (lat<80) & (lon>-180+360) & (lon<-100+360)))

    ### remove caspian sea, black sea, mediterranean sea, and baltic sea
    data = data.where(~((lat>24) & (lat<70) & (lon>14) & (lon<70)))
    
    ### remove hudson bay
    data = data.where(~((lat>50) & (lat<70) & (lon>-100+360) & (lon<-70+360)))
    data = data.where(~((lat>70) & (lat<80) & (lon>-130+360) & (lon<-80+360)))
    
    ### Remove Red sea
    data = data.where(~((lat>10) & (lat<25) & (lon>10) & (lon<45)))
    data = data.where(~((lat>20) & (lat<50) & (lon>0) & (lon<20)))
    
    return data


def load_socat_mask():
    '''
    load a mask of SOCAT data product
    '''
    # location of masked data
    dir_mask = f'{root_dir}/../data/artemis/observations/masks/processed'
    
    # load data with xarray
    ds_mask = xr.open_dataset(f'{dir_mask}/SOCATv2019_mask_1x1_198201-201512.nc')
    
    return ds_mask

#===============================================
# Data prep functions
#===============================================

def detrend_time(array_1d, N_time):
    """
        Input: 1d array and the length the time dimension should be
        Output: 1d array of original data less linear trend over time; any location with at least one nan is returned as nan for all times
        Method: assumes 2d array can be filled in column-wise (i.e. time was the first dimension that generated the 1d array)
    """
    array_2d = array_1d.reshape(N_time,-1,order='C')
    nan_mask = (np.any(np.isnan(array_2d), axis=0))
    X = np.arange( N_time )
    regressions = np.polyfit(X, array_2d[:,~nan_mask], 1)
    lin_fit = (np.expand_dims(X,1) * regressions[0:1,:] + regressions[1:,:])
    array_detrend_2d = np.empty(shape=array_2d.shape)
    array_detrend_2d[:] = np.nan
    array_detrend_2d[:,~nan_mask] = array_2d[:,~nan_mask] - lin_fit
    
    return array_detrend_2d.flatten(order='C')

def calc_anom(array_1d, N_time, N_batch, array_mask0=None):
    """
        Input: 1d array, the length the time dimension should be, and the window for averaging
        Output: 1d array of original data less mean during that time period
        Method: assumes 2d array can be filled in C order (i.e. time was the first dimension that generated the 1d array)
        Note: can include an extra array to use to adjust for values that should be set to 0
    """
    array_2d = array_1d.copy()
    if array_mask0 is not None:
        nan_mask = np.isnan(array_2d)
        mask0 = np.nan_to_num(array_mask0, nan=-1.0) <= 0
        array_2d[mask0] = np.nan
    array_2d = array_2d.reshape(N_time,-1,order='C')

    for i in range(-(-N_time//N_batch)):
        avg_val = np.nanmean(array_2d[(i*N_batch):((i+1)*N_batch),:])
        array_2d[(i*N_batch):((i+1)*N_batch),:] = array_2d[(i*N_batch):((i+1)*N_batch),:] - avg_val
    
    output = array_2d.flatten(order='C')
    if array_mask0 is not None:
        output[~nan_mask & mask0] = 0
    
    return output

def log_or_0(array_1d):
    """
        Input: 1d array
        Output: log of 1d array or 0 for values <=0
    """
    output_ma, output = array_1d.copy(), array_1d.copy()
    output_ma = np.ma.masked_array(output_ma, np.isnan(output_ma))
    output_ma = np.ma.log10(output_ma)
    output[~output_ma.mask] = output_ma[~output_ma.mask]
    output[output_ma.mask] = np.maximum(output[output_ma.mask],0)
    return output

#===============================================
# Loading in data and creating features
#===============================================

def import_member_data(ensemble_dir_head, ens, member, dates, xco2_path=f"{root_dir}/../models/CESM001/member_001/XCO2_1D_mon_CESM001_native_198201-201701.nc"):
    
    member_dir = f'{ensemble_dir_head}/CanESM2r1r10/member_r1r10'
    if ens == "CanESM2":
        # CanESM2 files are mislabeled as going to 201712
        
        sss_path = f"{member_dir}/SSS_2D_mon_CanESM2r1r10_1x1_198201-201712.nc"
        sst_path = f"{member_dir}/SST_2D_mon_CanESM2r1r10_1x1_198201-201712.nc"
        chl_path = f"{member_dir}/Chl_2D_mon_CanESM2r1r10_1x1_198201-201712.nc"
        mld_path = f"{member_dir}/MLD_2D_mon_CanESM2r1r10_1x1_198201-201712.nc"
        pco2_path = f"{member_dir}/pCO2_2D_mon_CanESM2r1r10_1x1_198201-201712.nc"
    
    else:
        sss_path = f"{member_dir}/SSS_2D_mon_{ens}{member}_1x1_198201-201701.nc"
        sst_path = f"{member_dir}/SST_2D_mon_{ens}{member}_1x1_198201-201701.nc"
        chl_path = f"{member_dir}/Chl_2D_mon_{ens}{member}_1x1_198201-201701.nc"
        mld_path = f"{member_dir}/MLD_2D_mon_{ens}{member}_1x1_198201-201701.nc"
        pco2_path = f"{member_dir}/pCO2_2D_mon_{ens}{member}_1x1_198201-201701.nc"

    inputs = {}

    inputs['sss'] = xr.open_dataset(sss_path).SSS
    inputs['sst'] = xr.open_dataset(sst_path).SST
    inputs['chl'] = xr.open_dataset(chl_path).Chl
    inputs['mld'] = xr.open_dataset(mld_path).MLD
    inputs['pco2'] = xr.open_dataset(pco2_path).pCO2
    inputs['xco2'] = xr.open_dataset(xco2_path).XCO2
    inputs['socat_mask'] = xr.open_dataset(sss_path).socat_mask

    for i in inputs:
        time_len = len(inputs[i].time.values)
        inputs[i] = inputs[i].assign_coords(time = dates[0:time_len])
        #inputs[i].time.values = dates[0:time_len]
        if i != 'xco2':
            inputs[i] = inputs[i].transpose('time', 'ylat', 'xlon')

    DS = xr.merge([inputs['sss'], inputs['sst'], inputs['mld'], inputs['chl'], inputs['pco2'], inputs['socat_mask']], compat='override')

    return DS, inputs['xco2']

def create_features(df, N_time=421, N_batch = 12):

    df['mld_log'] = log_or_0(df['MLD'].values)
    df['mld_anom'] = calc_anom(df['mld_log'].values, N_time, N_batch, array_mask0=df['MLD'].values)
    
    df_mld = df.loc[(df['MLD']>0),'MLD']
    mld_grouped = df_mld.groupby(by=[df_mld.index.get_level_values('time').month, 'xlon','ylat']).mean()
    df = df.join(mld_grouped, on = [df.index.get_level_values('time').month, 'xlon','ylat'], rsuffix="_clim")
    df['mld_clim_log'] = log_or_0(df['MLD_clim'].values)

    df['chl_log'] = log_or_0(df['Chl'].values)
    df['chl_anom'] = calc_anom(df['chl_log'].values, N_time, N_batch, array_mask0=(df['Chl'].values))

    df.rename(columns={'SSS':'sss'}, inplace=True)
    df['sss_anom'] = calc_anom(df['sss'].values, N_time, N_batch)

    df['sst_detrend'] = detrend_time(df['SST'].values, N_time)
    df['sst_anom'] = calc_anom(df['sst_detrend'].values, N_time, N_batch)

    days_idx = df.index.get_level_values('time').dayofyear.to_numpy()
    lon_rad = np.radians(df.index.get_level_values('xlon').to_numpy())
    lat_rad = np.radians(df.index.get_level_values('ylat').to_numpy())
    df['T0'], df['T1'] = [np.cos(days_idx * 2 * np.pi / 365), np.sin(days_idx * 2 * np.pi / 365)]
    df['A'], df['B'], df['C'] = [np.sin(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), -np.cos(lon_rad)*np.cos(lat_rad)]

    return df

def create_inputs(ensemble_dir_head, ens, member, dates, N_batch = 12, xco2_path=f"{root_dir}/../data/artemis/simulations/LET/CESM/member_001/XCO2_1D_mon_CESM001_native_198201-201701.nc"):
    
    print('start to import member data')
    DS, DS_xco2 = import_member_data(ensemble_dir_head, ens, member, dates, xco2_path=xco2_path)
    
    print('finish importing member data')
    df = DS.to_dataframe()
    
    print('start to create feature')
    df = create_features(df, N_time=len(dates), N_batch = N_batch) 
    
    net_mask = np.tile(network_mask().transpose('lon','lat').to_dataframe()['mask'].to_numpy(), len(dates))
    df['net_mask'] = net_mask
    df['xco2'] = np.repeat(DS_xco2.values, 360*180)

    return df


#===============================================
# Evaluation functions
#===============================================

def centered_rmse(y,pred):
    y_mean = np.mean(y)
    pred_mean = np.mean(pred)
    return np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size)

def evaluate_test(y, pred):
    scores = {
        'mse':mean_squared_error(y, pred),
        'mae':mean_absolute_error(y, pred),
        'medae':median_absolute_error(y, pred),
        'max_error':max_error(y, pred),
        'bias':pred.mean() - y.mean(),
        'r2':r2_score(y, pred),
        'corr':np.corrcoef(y,pred)[0,1],
        'cent_rmse':centered_rmse(y,pred),
        'stdev' :np.std(pred),
        'amp_ratio':(np.max(pred)-np.min(pred))/(np.max(y)-np.min(y)), # added when doing temporal decomposition
        'stdev_ref':np.std(y),
        'range_ref':np.max(y)-np.min(y),
        'iqr_ref':np.subtract(*np.percentile(y, [75, 25]))
        }
    return scores

#===============================================
# Train test split functions
#===============================================

def train_val_test_split(N, test_prop, val_prop):
    intermediate_idx, test_idx = train_test_split(range(N), test_size=test_prop)
    train_idx, val_idx = train_test_split(intermediate_idx, test_size=val_prop/(1-test_prop))
    return intermediate_idx, train_idx, val_idx, test_idx

def apply_splits(X, y, train_val_idx, train_idx, val_idx, test_idx):
    X_train_val = X[train_val_idx,:]
    X_train = X[train_idx,:]
    X_val = X[val_idx,:]
    X_test = X[test_idx,:]

    y_train_val = y[train_val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test

def cross_val_splits(train_val_idx, random_seeds, row, ens_count, folds=3):
    """Didn't actually use this"""
    idx = train_val_idx.copy()
    np.random.seed(random_seeds[row,ens_count])
    np.random.shuffle(idx)
    list_val = np.array_split(idx, folds)
    list_train = []
    for i in range(folds):
        list_train.append( np.concatenate(list_val[:i] + list_val[(i+1):]) )
    return zip(list_train, list_val)

#===============================================
# NN functions
#===============================================

def build_nn(num_features, neurons=[512,256], act='relu', use_drop=True, drop_rate=0.5, learning_rate=0.01, reg=0.001):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    for i in range(len(neurons)):
        model.add(Dense(units=neurons[i], activation=act, kernel_regularizer=regularizers.l2(reg)))
        if use_drop:
            model.add(Dropout(drop_rate))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model

def build_nn_vf(num_features, act='relu', learning_rate=0.01, reg=0.001):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_features,)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=500, activation=act, kernel_regularizer=regularizers.l2(reg)))
    model.add(Dense(units=1))

    model.compile(keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse'])

    return model


#===============================================
# Saving functions
#===============================================

def save_clean_data(df, data_output_dir, ens, member):
    print("Starting data saving process")
    output_dir = f"{data_output_dir}/{ens}/member_{member}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{output_dir}/data_clean_2D_mon_{ens}_{member}_1x1_198201-201701.pkl"
    df.to_pickle(fname)
    print("Save complete")
    

def save_model(model, model_output_dir, approach, ens, member, run=None):
    print("Starting model saving process")
    model_dir = f"{model_output_dir}/{approach}/{ens}/member_{member}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    if approach == 'nn':
        if run is None:
            run = 0
        model_fname = f"{model_dir}/{approach}_model_pC02_2D_mon_{ens}_{member}_{run}_1x1_198201-201701.h5"
        model.save(model_fname)
    else:
        model_fname = f"{model_dir}/{approach}_model_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.joblib"
        joblib.dump(model, model_fname)
    print("Save complete")

def save_recon(DS_recon, recon_output_dir, approach, ens, member, run=None):
    print("Starting reconstruction saving process")
    recon_dir = f"{recon_output_dir}/{approach}/{ens}/member_{member}"
    Path(recon_dir).mkdir(parents=True, exist_ok=True)
    if approach == "nn":
        if run is None:
            run = 0
        recon_fname = f"{recon_dir}/{approach}_recon_pC02_2D_mon_{ens}_{member}_{run}_1x1_198201-201701.nc"
    else:
        recon_fname = f"{recon_dir}/{approach}_recon_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.nc"
    DS_recon.to_netcdf(recon_fname)
    print("Save complete")


#===============================================
# Temporal deconstruction functions
#===============================================

def detrend_time_2d(array_1d, N_time):
    """
        Input: 1d array and the length the time dimension should be
        Output: 2d array of original data less linear trend over time
        Method: assumes 2d array can be filled in column-wise (i.e. time was the first dimension that generated the 1d array)
    """
    array_2d = array_1d.reshape(N_time,-1,order='C')
    nan_mask = (np.all(np.isnan(array_2d), axis=0))
    X = np.arange( N_time )
    try:
        regressions = np.polyfit(X, array_2d[:,~nan_mask], 1)
    except:
        regressions = np.empty((2,np.sum(~nan_mask)))
        j = 0
        for i in range(nan_mask.shape[0]):
            if ~nan_mask[i]:
                regress = np.polyfit(X[~np.isnan(array_2d[:,i])], array_2d[~np.isnan(array_2d[:,i]),i], 1)
                regressions[:,j] = regress
                j +=1
    lin_fit = (np.expand_dims(X,1) * regressions[0:1,:] + regressions[1:,:])
    array_detrend_2d = np.ones(shape=array_2d.shape)*np.nan
    array_detrend_2d[:,~nan_mask] = array_2d[:,~nan_mask] - lin_fit
    
    return array_detrend_2d

def calc_seasonal(array_2d, N_time=421, period=12):
    nan_mask = ~(np.all(np.isnan(array_2d), axis=0))
    month_avgs = np.array([np.nanmean(array_2d[i::period,nan_mask], axis=0) for i in range(period)])
    month_avgs_centered = month_avgs - np.nanmean(month_avgs, axis=0)
    
    seasonal = np.ones(array_2d.shape)*np.NaN
    seasonal[:,nan_mask] = np.tile(month_avgs_centered, [N_time // period + 1,1])[:N_time]
    
    return seasonal
                     

def apply_lowess(array_2d, x, frac_lo, it_lo, delta_lo):
    nan_mask = ~(np.all(np.isnan(array_2d), axis=0))
    smoothed = np.apply_along_axis(lowess, 0, array_2d[:,nan_mask],
                               exog=x, frac=frac_lo, it=it_lo, delta=delta_lo, 
                               is_sorted=True, missing="drop", return_sorted=False)
    out = np.ones(array_2d.shape)*np.NaN
    out[:,nan_mask] = smoothed
    
    return out


def detrend(y, N_time, period, x, frac_lo, frac_resid_lo, it_lo, delta_lo):
    y_detrend = detrend_time_2d(y, N_time)
    y_seasonal = calc_seasonal(y_detrend, N_time, period)
    y_deseason = y_detrend - y_seasonal

    # y_decadal = apply_lowess(y_deseason, x, frac_lo, it_lo, delta_lo)
    # y_resid = y_deseason - y_decadal
    # y_resid_lo = apply_lowess(y_resid, x, frac_resid_lo, it_lo, delta_lo)
    
    # add return of y_decadal, y_resid, and y_resid_lo if you want apply lowess
    return y_detrend, y_seasonal, y_deseason

def apply_detrend(approach, ens, member):

    N_time = 421
    period = 12
    x = np.arange(N_time)
    it_lo = 3
    delta_lo = 0.01 * N_time
    frac_lo = 12*10 / N_time
    frac_resid_lo = 12 / N_time

    recon_output_dir = f"{root_dir}/../models/reconstructions"
    print(root_dir)
    recon_dir = f"{recon_output_dir}/{approach}/{ens}/member_{member}"
            
    recon_fname = f"{recon_dir}/{approach}_recon_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.nc"
    recon_fname_out = f"{recon_dir}/{approach}_recon_temporal_pC02_2D_mon_{ens}_{member}_1x1_198201-201701.nc"

    DS_recon = xr.load_dataset(recon_fname)
    df = DS_recon.to_dataframe()

    y = df['pCO2'].values
    r = df['pCO2_recon'].values

    col_sel = ["net_mask", "socat_mask"]
    data_types = ["detrend", "seasonal", "deseason"]
    if approach == 'xg':
        # Only deconstruct the raw data for one of the reconstructions so don't need to duplicate computation
       
        y_output = detrend(y, N_time, period, x, frac_lo, frac_resid_lo, it_lo, delta_lo)
        col_sel.append("pCO2")
        for i in range(len(data_types)):
            df[f"pCO2_{data_types[i]}"] = y_output[i].flatten(order='C')
            col_sel.append(f"pCO2_{data_types[i]}")
    
    r_output = detrend(r, N_time, period, x, frac_lo, frac_resid_lo, it_lo, delta_lo)
    col_sel.append("pCO2_recon")
    for i in range(len(data_types)):
        df[f"pCO2_recon_{data_types[i]}"] = r_output[i].flatten(order='C')
        col_sel.append(f"pCO2_recon_{data_types[i]}")

    DS_recon = df[col_sel].to_xarray()
    DS_recon.to_netcdf(recon_fname_out)