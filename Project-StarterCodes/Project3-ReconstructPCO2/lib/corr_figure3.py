import numpy as np
import xarray as xr
import scipy.stats
import statsmodels.nonparametric.smoothers_lowess
from joblib import Parallel, delayed
import os
from scipy.signal import savgol_filter

def _detrend_ufunc_1d(arr):
    if np.isnan(arr).all():
        return arr
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return arr
    x = np.arange(mask.sum())
    y = arr[mask]
    if np.std(y) == 0 or len(np.unique(y)) < 2:
        return arr
    reg = scipy.stats.linregress(x, y)
    yfit = reg.intercept + reg.slope * x
    detrended = np.copy(arr)
    detrended[mask] = y - yfit
    return detrended
    
def process_detrend_column(arr2d, i):
    return _detrend_ufunc_1d(arr2d[:, i])

def detrend_fast_parallel(data, dim='time', n_jobs=-1):
    arr = data.values
    axis = data.get_axis_num(dim)
    arr = np.moveaxis(arr, axis, 0)
    original_shape = arr.shape
    arr2d = arr.reshape(arr.shape[0], -1)

    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_detrend_column)(arr2d, i) for i in range(arr2d.shape[1])
    )
    out2d = np.column_stack(results)
    out_arr = out2d.reshape(original_shape)
    out_arr = np.moveaxis(out_arr, 0, axis)
    return xr.DataArray(out_arr, dims=data.dims, coords=data.coords)

def _lowess_ufunc_1d(arr, lo_frac, lo_delta, it):
    if np.isnan(arr).all():
        return arr
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return arr
    x = np.arange(len(arr))[mask]
    y = arr[mask]
    trend = statsmodels.nonparametric.smoothers_lowess.lowess(
        y, x, frac=lo_frac, delta=lo_delta, it=it
    )
    result = np.full_like(arr, np.nan, dtype=np.float64)
    result[mask] = trend[:, 1]
    return result



def savgol_smoothing(data, window_length=11, polyorder=2, dim='time'):
    """
    Applies a Savitzky–Golay filter along the specified dimension of an xarray DataArray.
    
    Parameters:
      data : xarray.DataArray
          The input data.
      window_length : int
          The length of the filter window (must be odd). It defines how many points are used in the smoothing.
      polyorder : int
          The order of the polynomial used to fit the samples.
      dim : str
          The dimension along which to apply the filter.
          
    Returns:
      xarray.DataArray
          The smoothed data with the same coordinates and dimensions as the input.
    """
    # Extract the underlying numpy array and determine the axis to smooth
    arr = data.values
    axis = data.get_axis_num(dim)
    
    # Move the smoothing dimension to axis=0
    arr = np.moveaxis(arr, axis, 0)
    
    # Adjust window_length if it's larger than the data length along the smoothing dimension
    n = arr.shape[0]
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    
    # Apply savgol_filter along axis 0 (the moved dimension)
    smoothed = savgol_filter(arr, window_length=window_length, polyorder=polyorder, axis=0)
    
    # Move the axis back to its original position
    smoothed = np.moveaxis(smoothed, 0, axis)
    
    # Return a new DataArray with the same dims and coords as the input
    return xr.DataArray(smoothed, dims=data.dims, coords=data.coords)



def _seasonal_cycle_ufunc_1d(arr, period):
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    period_avgs = []
    for i in range(period):
        sub_arr = arr[i::period]
        if len(sub_arr) == 0 or np.isnan(sub_arr).all():
            period_avgs.append(np.nan)
        else:
            period_avgs.append(np.nanmean(sub_arr))
    period_avgs = np.array(period_avgs)
    if np.isnan(period_avgs).all():
        return np.full_like(arr, np.nan, dtype=np.float64)
    period_avgs = period_avgs - np.nanmean(period_avgs)
    repeated = np.tile(period_avgs, int(np.ceil(len(arr)/period)))[:len(arr)]
    return repeated

def process_season_column(arr2d, i, period):
    return _seasonal_cycle_ufunc_1d(arr2d[:, i], period)

def seasonal_cycle_fast_parallel(data, dim='time', period=12, n_jobs=-1):
    arr = data.values
    axis = data.get_axis_num(dim)
    arr = np.moveaxis(arr, axis, 0)
    original_shape = arr.shape
    arr2d = arr.reshape(arr.shape[0], -1)
    
    def process_column(i):
        return _seasonal_cycle_ufunc_1d(arr2d[:, i], period)
    # # 
    # if n_jobs == -1:
    #     n_jobs = max(1, os.cpu_count())

    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_season_column)(arr2d, i, period) for i in range(arr2d.shape[1])
    )
    # results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(process_column)(i) for i in range(arr2d.shape[1]))
    # results = Parallel(n_jobs=n_jobs)(delayed(process_column)(i) for i in range(arr2d.shape[1]))

    out2d = np.column_stack(results)
    out_arr = out2d.reshape(original_shape)
    out_arr = np.moveaxis(out_arr, 0, axis)
    return xr.DataArray(out_arr, dims=data.dims, coords=data.coords)

def decompose_stl_fast_parallel(da, var_name, lo_frac=0.1, lo_delta=0.01, it=3, period=12, n_jobs=-1):
    data = xr.open_zarr(da, consolidated=True)
    data = data.dropna(dim="time", how="all")
    data_detrend = detrend_fast_parallel(data[var_name], dim='time', n_jobs=n_jobs)
    data_seasonal = seasonal_cycle_fast_parallel(data_detrend, dim='time', period=period, n_jobs=n_jobs)
    data_deseason = data_detrend - data_seasonal
    data_lowess = savgol_smoothing(data_deseason, dim='time')
    data_residual = data_deseason - data_lowess
    data_residual_low = savgol_smoothing(data_residual, dim='time')
        
    var_name_spco2 = 'spco2'
    ds_out = xr.Dataset(
        {
            f'{var_name_spco2}': data[var_name],
            f'{var_name_spco2}_detrend': data_detrend,
            f'{var_name_spco2}_dec': data_lowess,
            f'{var_name_spco2}_seasonal': data_seasonal,
            f'{var_name_spco2}_residual': data_residual,
            f'{var_name_spco2}_residual_low': data_residual_low,
        },
        coords={'time': data['time'], 'ylat': data['ylat'], 'xlon': data['xlon']}
    )
    return ds_out

def correlation(a, b, axis=0):
    with np.errstate(invalid='ignore', divide='ignore'):
        count_valid = np.sum(~np.isnan(a) & ~np.isnan(b), axis=axis) 
        valid_mask = count_valid > 1 
        if np.all(count_valid == 0):  
            return np.full(a.shape[1:], np.nan)

        a_mean = np.full_like(a[0], np.nan, dtype=np.float64)
        b_mean = np.full_like(b[0], np.nan, dtype=np.float64)
        cov = np.full_like(a[0], np.nan, dtype=np.float64)

        valid_indices = np.where(valid_mask)
        a_mean[valid_indices] = np.nanmean(a[:, valid_indices[0], valid_indices[1]], axis=axis)
        b_mean[valid_indices] = np.nanmean(b[:, valid_indices[0], valid_indices[1]], axis=axis)

        cov[valid_indices] = np.nanmean((a[:, valid_indices[0], valid_indices[1]] - a_mean[valid_indices]) * 
                                        (b[:, valid_indices[0], valid_indices[1]] - b_mean[valid_indices]), axis=axis)

        std_a = np.full_like(a[0], np.nan, dtype=np.float64)
        std_b = np.full_like(b[0], np.nan, dtype=np.float64)

        valid_std = count_valid > 1
        std_a[valid_std] = np.nanstd(a[:, valid_std], axis=axis, ddof=0)
        std_b[valid_std] = np.nanstd(b[:, valid_std], axis=axis, ddof=0)

        valid_std_mask = (std_a > 0) & (std_b > 0)
        result = np.where(valid_std_mask, cov / (std_a * std_b), np.nan)

    return result

def std_star(a, b, axis=0):
    with np.errstate(invalid='ignore', divide='ignore'):
        count_valid = np.sum(~np.isnan(a) & ~np.isnan(b), axis=axis)
        valid_mask = count_valid > 1  
        
        if np.all(count_valid == 0):   
            return np.full(a.shape[1:], np.nan)

        std_a = np.full_like(a[0], np.nan, dtype=np.float64)
        std_b = np.full_like(b[0], np.nan, dtype=np.float64)

        std_a[valid_mask] = np.nanstd(a[:, valid_mask], axis=axis, ddof=0)
        std_b[valid_mask] = np.nanstd(b[:, valid_mask], axis=axis, ddof=0)

        valid_std = (std_b > 0)
        result = np.where(valid_std, std_a / std_b, np.nan)

    return result

def calculate_GCB_statistics_fast(ds_mod, ds_som):
    ds_mod = ds_mod.compute()
    ds_som = ds_som.compute()
    
    corr_detrend = correlation(ds_mod['spco2_detrend'].values, ds_som['spco2_detrend'].values, axis=0)
    corr_dec = correlation(ds_mod['spco2_dec'].values, ds_som['spco2_dec'].values, axis=0)
    corr_seasonal = correlation(ds_mod['spco2_seasonal'].values, ds_som['spco2_seasonal'].values, axis=0)
    corr_residual = correlation(ds_mod['spco2_residual'].values, ds_som['spco2_residual'].values, axis=0)
    corr_residual_low = correlation(ds_mod['spco2_residual_low'].values, ds_som['spco2_residual_low'].values, axis=0)

    std_star_all = std_star(ds_mod['spco2'].values, ds_som['spco2'].values, axis=0)
    std_star_detrend = std_star(ds_mod['spco2_detrend'].values, ds_som['spco2_detrend'].values, axis=0)
    std_star_dec = std_star(ds_mod['spco2_dec'].values, ds_som['spco2_dec'].values, axis=0)
    std_star_seasonal = std_star(ds_mod['spco2_seasonal'].values, ds_som['spco2_seasonal'].values, axis=0)
    std_star_residual = std_star(ds_mod['spco2_residual'].values, ds_som['spco2_residual'].values, axis=0)
    std_star_residual_low = std_star(ds_mod['spco2_residual_low'].values, ds_som['spco2_residual_low'].values, axis=0)
    
    ds_out = xr.Dataset(
        {
            'corr_detrend': (['ylat', 'xlon'], corr_detrend),
            'corr_dec': (['ylat', 'xlon'], corr_dec),
            'corr_seasonal': (['ylat', 'xlon'], corr_seasonal),
            'corr_residual': (['ylat', 'xlon'], corr_residual),
            'corr_residual_low': (['ylat', 'xlon'], corr_residual_low),
            'std_star': (['ylat', 'xlon'], std_star_all),
            'std_star_detrend': (['ylat', 'xlon'], std_star_detrend),
            'std_star_dec': (['ylat', 'xlon'], std_star_dec),
            'std_star_seasonal': (['ylat', 'xlon'], std_star_seasonal),
            'std_star_residual': (['ylat', 'xlon'], std_star_residual),
            'std_star_residual_low': (['ylat', 'xlon'], std_star_residual_low),
        },
        coords={
            'ylat': (['ylat'], ds_mod['ylat'].values),
            'xlon': (['xlon'], ds_mod['xlon'].values)
        }
    )
    return ds_out

def eval_spatial(selected_mems_dict, recon_output_dir, init_date, fin_date):

    ds_ens_list = []  
    for ens, members in selected_mems_dict.items():
        ds_mem_list = [] 
        for member in members:
            print(f"Starting computation for ESM: {ens}, Member: {member}") 
            recon_dir = f"{recon_output_dir}/{ens}/{member}"
            recon_path = f"{recon_dir}/recon_pCO2_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
            
            detrend_recon = decompose_stl_fast_parallel(da=recon_path, var_name="pCO2_recon_unseen")
            detrend_truth = decompose_stl_fast_parallel(da=recon_path, var_name="pCO2_truth")
    
            ds_eval = calculate_GCB_statistics_fast(ds_mod=detrend_truth, ds_som=detrend_recon)
            ds_eval = ds_eval.expand_dims({"member": [member]})
            ds_mem_list.append(ds_eval)
    
        ds_ens = xr.concat(ds_mem_list, dim="member")
        ds_ens = ds_ens.expand_dims({"ens": [ens]})
        ds_ens_list.append(ds_ens)
    
    ds_final = xr.concat(ds_ens_list, dim="ens")
    return ds_final


    