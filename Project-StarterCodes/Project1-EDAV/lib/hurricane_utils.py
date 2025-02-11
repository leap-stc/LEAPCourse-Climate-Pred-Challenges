import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Tuple, List, Optional, Union

# import os
# import xarray as xr
# from numpy import linalg as LA
# from sklearn.cluster import KMeans

def map_background(label=False, extent=[-100, 0, 0, 60]):
  # A helpder function for creating the map background.
  # INPUT:
  # "extent": corresponds to the location information of the showed map.
  # "label": boolean

  # OUTPUT:
  # Matplotlib AXES object

  plt.figure(figsize = (20, 10))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()
  ax.set_extent(extent)
  ax.gridlines(draw_labels=label) # show labels or not
  LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                      edgecolor='face',
                                      facecolor=cfeature.COLORS['land'],
                                          linewidth=.1)
  OCEAN = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                       edgecolor='face',
                                       facecolor=cfeature.COLORS['water'], linewidth=.1)
  ax.add_feature(LAND, zorder=0)
  ax.add_feature(OCEAN)
  return ax

# These are helper functions to conveniently plot one single track.
def get_lon_lat(storm):
  lon_lst = storm.lon.values
  lat_lst = storm.lat.values
  return lon_lst[~np.isnan(lon_lst)], lat_lst[~np.isnan(lat_lst)]

# !!! Note that even though it's a convention to place latitude before longitude,
# to work with cartopy projection, longitude MUST be placed first. !!!
def plot_one_track(ax, storm, color='blue', all_track=True, marker_color='black'):
  # INPUT:
  # ax: Matplotlib axes object
  # storm: a Xarray DATASET object (this can be adjusted if desired)
  # all_track: plot the whole track or just the origin

  # OUTPUT:
  # None

  lon_lst, lat_lst = get_lon_lat(storm)
  year = storm.season.values
  if all_track:
    ax.plot(lon_lst, lat_lst, '-o', color=color, linewidth=2, markersize=3) # marker='.'
    ax.plot(lon_lst[-1], lat_lst[-1], color=marker_color, marker='x', markersize=10)
  ax.plot(lon_lst[0], lat_lst[0], color=marker_color, marker='*', markersize=10)
  ax.text(lon_lst[0], lat_lst[0]-2, str(storm.name.values)[2:-1], horizontalalignment='center')

def get_moments(storm):
  # A function to calculate the track moments given a storm
  # OUTPUT:
  # X-centroid, Y-centroid, X_var, Y_var, XY_var

  # Note that:
  # In this case, no weights are set. In other words, all weights are 1.
  # A weight variable would need to be added in order to explore other weights

  lon_lst, lat_lst = get_lon_lat(storm)
  # If the track only has one point, there is no point in calculating the moments
  if len(lon_lst)<= 1: return None
      
  # M1 (first moment = mean). 
  # No weights applied
  lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
    
  # M2 (second moment = variance of lat and of lon / covariance of lat to lon
  # No weights applied
  cv = np.ma.cov([lon_lst, lat_lst])
    
  return [lon_weighted, lat_weighted, cv[0, 0], cv[1, 1], cv[0, 1]]

def get_weighted_moments(storm):
    # A function to calculate the track moments given a storm
    # OUTPUT:
    # X-centroid, Y-centroid, X_var, Y_var, XY_var

    # A weight variable would need to be added in order to explore other weights
    lon_lst, lat_lst = get_lon_lat(storm)
    # If the track only has one point, there is no point in calculating the moments
    if len(lon_lst)<= 1: return None
        
    # M1 (first moment = mean). 
    # No weights applied
    lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
    
    # M2 (second moment = variance of lat and of lon / covariance of lat to lon
    # No weights applied
    cv = np.ma.cov([lon_lst, lat_lst])
    
    # Apply scaling weights from paper:
    # Centroids (first 2) get multiplied by 0.5/2
    # Variance components (last 3) get multiplied by 0.5/3
    return [lon_weighted * 0.5/2, 
            lat_weighted * 0.5/2, 
            cv[0, 0] * 0.5/3, 
            cv[1, 1] * 0.5/3, 
            cv[0, 1] * 0.5/3]

def delta_lon_lat(storm):
  lon_lst, lat_lst = get_lon_lat(storm)
  return [np.ptp(lon_lst), np.ptp(lat_lst)]

def is_named_storm(status_array):
    """Check if storm ever reached named storm status (HU or TS)"""
    # Convert all statuses to regular strings and clean them
    statuses = [str(x).strip("b'") for x in status_array]
    # Check if either HU or TS appears in the status array
    return any(status in ['HU', 'TS'] for status in statuses)

def get_tropical_cyclones(tks, basin='NA', start_year=1950, verbose=True):
    """
    Filter tropical cyclones by basin and classification.
    
    Args:
        tks: IBTrACS dataset
        basin: 'NA' (North Atlantic), 'WP' (West Pacific), 'EP' (East Pacific), 
               or 'SP' (South Pacific)
        start_year: Starting year for analysis
        verbose: Whether to print summary statistics
    
    Returns:
        numpy array of boolean mask for valid storms
    """
    basin_names = {
        'NA': 'North Atlantic',
        'WP': 'West Pacific',
        'EP': 'East Pacific',
        'SP': 'South Pacific'
    }
    
    # Define valid classifications for each basin
    classifications = {
        'NA': ['HU', 'TS'],  # Atlantic hurricanes and tropical storms
        'WP': ['TY', 'TS'],  # Typhoons and tropical storms
        'EP': ['HU', 'TS'],  # Eastern Pacific hurricanes and tropical storms
        'SP': ['TC', 'TS']   # Tropical cyclones and tropical storms
    }

    # Inner function!
    def is_valid_storm(status_array):
        """Check if storm reached valid classification for given basin"""
        statuses = [str(x).strip("b'") for x in status_array]
        return any(status in classifications[basin] for status in statuses)
    
    # Create masks
    basin_mask = np.array([str(x[0]) == f"b'{basin}'" for x in tks.basin.values])
    named_storm_mask = np.array([is_valid_storm(tks.usa_status[i].values) 
                                for i in range(tks.dims['storm'])])
    year_mask = (tks.season >= start_year).values
    valid_storms = np.logical_and.reduce([named_storm_mask, year_mask, basin_mask])
    
    if verbose:
        print(f"\n{basin_names[basin]} Basin Analysis:")
        print("-" * 30)
        print(f"Total storms in dataset: {tks.dims['storm']}")
        print(f"Storms in {basin_names[basin]}: {np.sum(basin_mask)}")
        print(f"Named storms/typhoons: {np.sum(named_storm_mask)}")
        print(f"Storms after {start_year}: {np.sum(year_mask)}")
        print(f"Storms passing all filters: {np.sum(valid_storms)}")
        
        # Additional statistics
        if np.sum(valid_storms) > 0:
            valid_years = tks.season.values[valid_storms]
            print(f"\nTime period covered: {int(np.min(valid_years))} - {int(np.max(valid_years))}")
            print(f"Average storms per year: {np.sum(valid_storms)/(np.max(valid_years)-np.min(valid_years)+1):.1f}")
    
    return valid_storms

def analyze_clusters(tks, labels: np.ndarray, valid_storms: np.ndarray) -> Tuple[plt.Figure, dict]:
    """
    Analyze hurricane clusters and create violin plots for various characteristics.
    
    Parameters:
    -----------
    tks : xarray.Dataset
        IBTrACS dataset containing hurricane data
    labels : np.ndarray
        Cluster labels for each storm
    valid_storms : np.ndarray
        Boolean array indicating valid storms
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the violin plots
    stats : dict
        Dictionary containing summary statistics for each cluster
    """
    # Get indices of valid storms
    valid_indices = np.where(valid_storms)[0]
    
    # Initialize lists to store data for all clusters
    all_max_winds = []  # Maximum sustained wind speed
    all_lifespans = []  # Storm duration in days
    all_pdis = []      # Power Dissipation Index
    all_lengths = []   # Track length
    cluster_labels = []
    
    num_clusters = len(np.unique(labels))
    storm_counts = np.zeros(num_clusters)
    
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        storm_counts[i] = len(cluster_indices)
        
        for idx in cluster_indices:
            storm_idx = valid_indices[idx]
            storm = tks.isel(storm=storm_idx)
            
            # Maximum wind speed (using USA_WIND as it's 1-min sustained wind)
            winds = storm.usa_wind.values
            winds = winds[~np.isnan(winds)]
            max_wind = np.max(winds) if len(winds) > 0 else np.nan
            
            # Lifespan (number of 6-hourly observations converted to days)
            lifespan = np.sum(~np.isnan(storm.lon.values)) * 6 / 24
            
            # Calculate PDI (integrated cube of maximum wind speed)
            pdi = np.sum(np.power(winds[winds > 0], 3)) * 6 * 3600  # Convert 6-hourly to seconds
            
            # Track length (great circle distance)
            lons = storm.lon.values[~np.isnan(storm.lon.values)]
            lats = storm.lat.values[~np.isnan(storm.lat.values)]
            if len(lons) > 1:
                length = calculate_track_length(lons, lats)
            else:
                length = 0
                
            if not np.isnan(max_wind):  # Only include storms with valid data
                all_max_winds.append(max_wind)
                all_lifespans.append(lifespan)
                all_pdis.append(pdi)
                all_lengths.append(length)
                cluster_labels.append(f'Cluster {i+1}')
    
    # Create storm count plot and characteristic violin plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot storm counts
    axes[0].bar(range(1, num_clusters + 1), storm_counts)
    axes[0].set_title('Number of Storms per Cluster')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Storms')
    axes[0].grid(True, alpha=0.3)
    
    # Plot characteristics
    data_pairs = [
        (all_max_winds, 'Maximum Wind Speed', 'Wind Speed (knots)'),
        (all_lifespans, 'Life Span', 'Duration (days)'),
        (all_pdis, 'Power Dissipation Index', 'PDI (m³/s³)'),
        (all_lengths, 'Track Length', 'Length (km)')
    ]
    
    for idx, (data, title, ylabel) in enumerate(data_pairs):
        sns.violinplot(x=cluster_labels, y=data, ax=axes[idx+1])
        axes[idx+1].set_title(title)
        axes[idx+1].set_xlabel('Cluster')
        axes[idx+1].set_ylabel(ylabel)
        axes[idx+1].tick_params(axis='x', rotation=45)
    
    # Plot seasonal distribution
    # Collect months for each cluster
    cluster_months = {i: [] for i in range(num_clusters)}
    all_months = []
    
    for label_idx, storm_idx in enumerate(valid_indices):
        cluster = labels[label_idx]
        storm = tks.isel(storm=storm_idx)
        
        try:
            time_vals = storm.time.values
            time_vals = time_vals[~np.isnan(time_vals)]
            if len(time_vals) > 0:
                time = pd.Timestamp('1858-11-17') + pd.Timedelta(hours=float(time_vals[0]))
                month = time.month
                cluster_months[cluster].append(month)
                all_months.append(month)
        except (AttributeError, IndexError, ValueError) as e:
            print(f"Warning: Could not process time for storm {storm_idx}: {e}")
            continue
    
    # Prepare data for box plot
    plot_data = []
    labels_for_plot = []
    
    # Add data for each cluster
    for i in range(num_clusters):
        if cluster_months[i]:
            plot_data.append(cluster_months[i])
            labels_for_plot.append(str(i+1))
    
    # Add "All" category
    plot_data.append(all_months)
    labels_for_plot.append('All')
    
    # Create box plot in the last subplot
    bp = axes[5].boxplot(plot_data, patch_artist=False, showfliers=True)
    
    # Customize seasonal plot
    axes[5].set_ylabel('Month')
    axes[5].set_xlabel('Cluster')
    axes[5].set_title('Seasonal Distribution')
    
    # Set y-axis ticks and labels
    axes[5].set_yticks(range(1, 13))
    axes[5].set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Set x-axis labels
    axes[5].set_xticklabels([f'Cluster {x}' if x != 'All' else x for x in labels_for_plot])
    axes[5].tick_params(axis='x', rotation=45)
    
    # Add grid for easier reading
    axes[5].yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits with some padding
    axes[5].set_ylim(0.5, 12.5)
    
    plt.tight_layout()
    
    # Calculate summary statistics
    stats = {}
    for i in range(num_clusters):
        cluster_mask = np.array(cluster_labels) == f'Cluster {i+1}'
        stats[f'Cluster {i+1}'] = {
            'n_storms': np.sum(cluster_mask),
            'avg_wind': np.mean(np.array(all_max_winds)[cluster_mask]),
            'avg_lifespan': np.mean(np.array(all_lifespans)[cluster_mask]),
            'avg_pdi': np.mean(np.array(all_pdis)[cluster_mask]),
            'avg_length': np.mean(np.array(all_lengths)[cluster_mask])
        }
    
    return fig, stats

def calculate_track_length(lons: np.ndarray, lats: np.ndarray) -> float:
    """
    Calculate the total track length in kilometers using great circle distances.
    
    Parameters:
    -----------
    lons : np.ndarray
        Array of longitudes
    lats : np.ndarray
        Array of latitudes
        
    Returns:
    --------
    float
        Total track length in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)
    
    # Calculate distances between consecutive points
    total_distance = 0
    for i in range(len(lons_rad)-1):
        dlat = lats_rad[i+1] - lats_rad[i]
        dlon = lons_rad[i+1] - lons_rad[i]
        
        a = np.sin(dlat/2)**2 + np.cos(lats_rad[i]) * np.cos(lats_rad[i+1]) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        total_distance += R * c
        
    return total_distance

def plot_cluster_trends(tks, labels, valid_storms):
    """
    Create plot showing cluster trends over time with 5-year moving average.
    
    Parameters:
    -----------
    tks : xarray.Dataset
        IBTrACS dataset
    labels : array-like
        Cluster labels for each storm
    valid_storms : array-like
        Boolean mask for valid storms
    """
    # Get valid storm indices and their years
    valid_indices = np.where(valid_storms)[0]
    years = [int(tks.season[i].values) for i in valid_indices]
    
    # Create DataFrame with year and cluster label
    # Add 1 to labels for 1-based indexing
    df = pd.DataFrame({
        'Year': years,
        'Cluster': labels + 1  # Add 1 to make clusters 1-based
    })
    
    # Count storms per cluster per year
    yearly_counts = df.pivot_table(
        index='Year', 
        columns='Cluster',
        aggfunc='size',
        fill_value=0
    )
    
    # Create line plot with moving average
    window = 5  # 5-year moving average
    moving_avgs = yearly_counts.rolling(window=window, center=True).mean()
    
    plt.figure(figsize=(12, 6))
    moving_avgs.plot(marker='.', markersize=4)
    plt.title(f'{window}-Year Moving Average of Storm Count by Cluster')
    plt.xlabel('Year')
    plt.ylabel('Number of Storms (Moving Average)')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print trend statistics
    print("\nLinear trends in storm counts (storms/decade):")
    for cluster in sorted(yearly_counts.columns):
        y = yearly_counts[cluster]
        x = np.array(range(len(y)))
        z = np.polyfit(x, y, 1)
        trend_per_decade = z[0] * 10  # Convert per-year to per-decade
        print(f"Cluster {cluster}: {trend_per_decade:.2f}")