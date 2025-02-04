# Project 2 - Dataset

Accessing the Dataset
To ensure efficient data usage and minimize redundant storage, the LEAP Pangeo shared server provides centralized access to the dataset. Instead of downloading multiple copies, you can directly load the dataset from the shared server using the following URLs:

Training Dataset:

```bash
import xarray as xr
store = 'https://nyu1.osn.mghpcc.org/leap-pangeo-manual/GOTM/sf_training_data.zarr'
d = xr.open_dataset(store, engine='zarr', chunks={})
```

Testing Data:
```bash
import numpy as np
url = "https://nyu1.osn.mghpcc.org/leap-pangeo-manual/GOTM/data_testing_4_paper.txt"
valid_data = np.loadtxt(url, delimiter=None)[:,3:] 
```

These methods allow you to load data directly into your project without consuming additional local storage. 

However, if there are any issues accessing the shared server, you can download the dataset from the provided alternative link. Note that the Project2_Starter.ipynb notebook is designed to work with the shared server method. If you choose to download the dataset instead, you will need to make some modifications to the notebook to ensure compatibility.



## Download `code_and_data.zip`
1. Visit the following link to download the dataset:
   [Zenodo Dataset Link](https://zenodo.org/records/8293998)

2. After downloading, extract the `code_and_data.zip` file.

## Extract Required Files
1. Navigate to the `Data` folder within the extracted files.
2. Copy the following files from the `Data` folder:
   - `training_data_raw_time_index_49_149.nc`
   - `data_testing_4_paper.txt`
   - `training_data_for_SF_hbl_gaps_filled.nc`

## Upload to Project Directory
1. Place the copied files into the `Data` folDer in your project directory.
2. Ensure the file structure looks like this:
```
   project_root/
   ├── data/
   │   ├── training_data_raw_time_index_49_149.nc
   │   ├── data_testing_4_paper.txt
   │   ├── training_data_for_SF_hbl_gaps_filled.nc
```
