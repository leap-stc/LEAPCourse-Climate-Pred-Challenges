'''
These functions help us reduce runtime in the final notebook for review.
All code was previously run and stored in Google Cloud. We access these
results in the final notebook using load_corr_results and load_eval_results.
'''

import xarray as xr
import pickle
import gcsfs
import pandas as pd
from lib.bias_figure2 import XarrayEvaluator
from lib.corr_figure3 import eval_spatial
import os

# Date range consistent with datasets
date_range_start = '2004-01-01T00:00:00.000000000'
date_range_end = '2023-12-31T00:00:00.000000000'
dates = pd.date_range(start=date_range_start, end=date_range_end, freq='MS')
init_date = str(dates[0].year) + format(dates[0].month,'02d')
fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')

# Models and members (consistent across approaches)
models = ['ACCESS-ESM1-5', 'CanESM5', 'MPI-ESM1-2-LR']
members = {
    'ACCESS-ESM1-5': ['member_r10i1p1f1', 'member_r2i1p1f1', 'member_r5i1p1f1'],
    'CanESM5': ['member_r1i1p1f1', 'member_r1i1p2f1', 'member_r2i1p1f1'],
    'MPI-ESM1-2-LR': ['member_r11i1p1f1', 'member_r12i1p1f1', 'member_r15i1p1f1']
}

base_path = 'gs://leap-persistent/dhruvc99/dhruvc99/pco2_residual'

approaches = [
    'baseline', 
    'custom_loss', 
    'latitude_split', 
    'ocean_split', 
    'transfer_learning', 
    'weighted_samples'
]

def load_datasets_for_approach(approach):
    """
    Load and concatenate datasets for a specific approach.
    
    Args:
        approach (str): Name of the approach (e.g., 'baseline')
        
    Returns:
        xr.Dataset: Concatenated dataset containing reconstructions and truth
    """
    print(f"Loading datasets for {approach}...")
    
    # Path to reconstructions for this approach
    recon_path = f"{base_path}/{approach}/post02_xgb/reconstructions"
    
    # Dictionary to store datasets by model
    datasets_by_model = {}
    
    for model in models:
        # Dictionary to store datasets by member
        datasets_by_member = []
        
        for member in members[model]:
            try:
                # Full path to reconstruction file
                member_dir = f"{recon_path}/{model}/{member}"
                recon_file = f"{member_dir}/recon_pCO2_{model}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
                
                # Check if file exists
                if fs.exists(recon_file):
                    # Load truth and reconstruction
                    truth = xr.open_zarr(recon_file, consolidated=True)["pCO2_truth"]
                    recon = xr.open_zarr(recon_file, consolidated=True)["pCO2_recon_unseen"]
                    
                    # Assign dimension coordinates
                    truth = truth.assign_coords(status='truth')
                    recon = recon.assign_coords(status='reconstructed')
                    
                    # Ensure same time coordinates
                    common_time = np.intersect1d(truth['time'], recon['time'])
                    truth = truth.sel(time=common_time)
                    recon = recon.sel(time=common_time)
                    
                    # Concatenate along status dimension
                    member_ds = xr.concat([truth, recon], dim='status')
                    
                    # Add member dimension
                    member_ds = member_ds.expand_dims({"member": [member]})
                    
                    datasets_by_member.append(member_ds)
                    print(f"  - Loaded {model}/{member}")
                else:
                    print(f"  - File not found: {recon_file}")
            except Exception as e:
                print(f"  - Error loading {model}/{member}: {e}")
        
        if datasets_by_member:
            # Concatenate members for this model
            model_ds = xr.concat(datasets_by_member, dim="member")
            
            # Add model dimension
            model_ds = model_ds.expand_dims({"ens": [model]})
            
            datasets_by_model[model] = model_ds
    
    if datasets_by_model:
        # Concatenate all models
        concat_ds = xr.concat(list(datasets_by_model.values()), dim='ens')
        return concat_ds
    else:
        print(f"No valid datasets found for {approach}")
        return None

# Dictionary to store evaluation results
eval_results = {}
corr_results = {}
approach_results = {}

# Process each approach
for approach in approaches:
    try:
        # Load datasets
        concat_ds = load_datasets_for_approach(approach)
        
        if concat_ds is not None:
            # Store dataset for later use
            approach_results[approach] = concat_ds
            
            # Compute bias, RMSE, and correlation
            evaluator = XarrayEvaluator(concat_ds)
            ds_eval = evaluator.compute_all_metrics()
            eval_results[approach] = ds_eval
            
            # Compute temporal correlations
            selected_mems_dict = {model: members[model] for model in models}
            recon_output_dir = f"{base_path}/{approach}/post02_xgb/reconstructions"
            
            try:
                ds_eval_corr = eval_spatial(selected_mems_dict, recon_output_dir, init_date, fin_date)
                corr_results[approach] = ds_eval_corr
                print(f"Successfully computed spatial correlations for {approach}")
            except Exception as e:
                print(f"Error computing spatial correlations for {approach}: {e}")
    
    except Exception as e:
        print(f"Error processing {approach}: {e}")

fs = gcsfs.GCSFileSystem()

# Create a directory to save results if it doesn't exist
output_dir = base_path+'/analysis_results'
if not fs.exists(output_dir):
    fs.mkdir(output_dir)

# Save eval_results as pickle
with fs.open(f'{output_dir}/eval_results.pkl', 'wb') as f:
    pickle.dump(eval_results, f)
    
# Save corr_results as pickle
with fs.open(f'{output_dir}/corr_results.pkl', 'wb') as f:
    pickle.dump(corr_results, f)

print(f"Results saved to {output_dir}")

# ACCESS PRERAN RESULTS. THESE ARE USED IN FINAL NOTEBOOK.
def load_corr_results():
    corr_path = 'gs://leap-persistent/dhruvc99/dhruvc99/pco2_residual/analysis_results/corr_results.pkl'
    with fs.open(corr_path, 'rb') as file:
        return pickle.load(file)

def load_eval_results():
    eval_path = 'gs://leap-persistent/dhruvc99/dhruvc99/pco2_residual/analysis_results/eval_results.pkl'
    with fs.open(eval_path, 'rb') as file:
        return pickle.load(file)
        