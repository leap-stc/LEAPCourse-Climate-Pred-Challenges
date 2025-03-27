# `lib/` - Helper Modules for ML-based pCO₂ Reconstruction

This folder contains helper scripts that support machine learning reconstruction of surface ocean pCO₂ fields, including data preparation, feature engineering, evaluation, and visualization.


## Files and Descriptions

### `residual_utils.py`
Core module for data handling and modeling. Includes:

- **Masking:** Removes land, coastal zones, and specific marginal seas from the ocean grid using topography and land-sea masks.
- **Anomaly Calculation:** Computes climatological and interannual anomalies for physical drivers like SST, SSS, MLD, and Chl-a.
- **Feature Engineering:** Constructs physical and spatial features for ML models (e.g., time harmonics, spatial sine/cosine terms).
- **Data Splitting:** Tools to partition time-series data into train, validation, and test sets.
- **Evaluation:** Computes performance metrics like bias, RMSE, R², correlation, and centered RMSE.
- **Saving & I/O:** Functions to save cleaned ML input data, trained models, and reconstructed fields (supports local and GCS outputs).
- **Post-processing:** Combines reconstructed residual pCO₂ with modeled temperature-driven pCO₂ (pCO₂-T) to compute total surface pCO₂ fields.


### `visualization.py`
Provides the `SpatialMap2` class for high-quality geospatial plotting using Cartopy and Matplotlib. Features include:

- Support for global and Southern Ocean projections.
- Clean subplot layouts with customizable colorbars.
- Optional land/coastline features and polar stereographic view.
- Includes `xr_add_cyclic_point()` for seamless longitude wrapping in maps.


### `bias_figure2.py`
- Loads reconstructed and truth pCO₂ datasets, computes reconstruction bias using `xarray`, and provides tools for ensemble-averaged spatial evaluation.
- Includes an `XarrayEvaluator` class for calculating bias, RMSE, and correlation across time.


### `corr_figure3.py`
Provides tools for **temporal decomposition** and **correlation analysis** of reconstructed surface pCO₂ fields compared to truth, following methods similar to *Gloege et al. (2020)*.

Key features:

- **STL-style decomposition**:
  - Linear detrending
  - Seasonal cycle extraction
  - Residual and smoothed components using Savitzky–Golay filtering
- **Parallelized computation** across spatial grids using `joblib`.
- **Evaluation metrics**:
  - Gridwise correlation (`correlation`)
  - Amplitude ratio (`std_star`)
- eval_spatial function runs full decomposition + metric computation across ensemble members.

---

## Usage

These modules are imported in the main notebook `notebooks/Project3_Starter.ipynb`, and can be reused or modified for similar ML applications in ocean and climate science.
