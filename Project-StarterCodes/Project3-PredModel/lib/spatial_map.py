import cmocean as cm
import cartopy.feature as cfeature
import scipy.io
import scipy
from scipy.stats import stats

import matplotlib.path as mpath
import numpy as np
import xarray as xr
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import matplotlib.pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

class SpatialMap(object):
    """
    SpatialMap : class to plot plot nice spatial maps with a colorbar 
                 correctly positioned in the figure
                 
    Inputs
    ==============
    data     : Input 2D dataset [lon,lat] (default=None)
    lon      : longitude vector (default=np.arange(0.5,360,1))
    lat      : latitude vector (default=np.arange(-89.5,90,1))
    region   : 'world', 'southern-ocean' (default='world')
    fig      : figure handle (default=None)
    rect     : number of rows, columns, and position (default=111)
    cmap     : colormap (default=cm.cm.balance)
    colorbar : Toggle for colorbar (default=True)
    ncolors  : number of colors in colorbar (default=101)
    vrange   : colorbar range (default=[0,1])
    
    Returns
    ==============
    returns a colormap of your data within the specified region 
    
    Methods
    ==============
    set_ticks()
    set_title()
    set_cbar_title()
    set_cbar_labels()

    Add at some point
    ==============
    # worldmap.cbar.ax.yaxis.set_ticks_position("left") # way to easily set tick location
    # worldmap.cbar.ax.yaxis.set_label_position('left') # set label position
    
    
    Example
    ==============
    # download WOA data
    ds = xr.open_dataset('https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa/salinity/decav/1.00/woa18_decav_s00_01.nc', decode_times=False)
    data = ds['s_mn'].where(ds['depth']==0, drop=True).mean(['time','depth'])
    # plot spatial map
    worldmap = SpatialMap(data, lon=ds['lon'], lat=ds['lat'], fig=plt.figure(figsize=(7,7)), vrange=[30, 37], region='world')
    
    """

    def __init__(self, 
                 nrows_ncols=(1, 1),
                 region='world', 
                 fig=None, 
                 rect=[1,1,1],  
                 colorbar=True, 
                 cbar_location='right',
                 cbar_mode='single',
                 cbar_orientation = 'vertical',
                 cbar_size='7%', 
                 cbar_pad=0.1, 
                 axes_pad = 0.2):
                 #cmap=cm.cm.balance,
                 #ncolors=101,
                 #vrange = [0, 1]):
        
        self.region = region
        self.cbar_orientation = cbar_orientation
        #self.vrange = vrange
        #self.ncolors = ncolors
        #self.cmap = cmap
        
        ### Setup figure and axes
        if fig is None:
            fig = plt.figure(figsize=(8.5,11))
            
        # Define projection
        if self.region.upper()=='SOUTHERN-OCEAN':
            projection = ccrs.SouthPolarStereo()
        
        if self.region.upper()=='WORLD':
            projection=ccrs.Robinson(central_longitude=-149.5)
            
        # Setup axesgrid
        axes_class = (GeoAxes, dict(map_projection=projection))
        self.grid = AxesGrid(fig, 
                             rect=rect, 
                             axes_class=axes_class,
                             share_all=False,
                             nrows_ncols = nrows_ncols,
                             axes_pad = axes_pad,
                             cbar_location = cbar_location,
                             cbar_mode= cbar_mode if colorbar==True else None,
                             cbar_pad = cbar_pad if colorbar==True else None,
                             cbar_size = cbar_size,
                             label_mode = '')  # note the empty label_mode 
    
    def add_plot(self, lon=None, lat=None, data=None, 
                 ax=None, 
                 land=True, 
                 coastline=True, 
                 linewidth_coast=0.25, 
                 ncolors=101, 
                 vrange=[0, 1], 
                 cmap=cm.cm.balance,
                 facecolor=[0.25,0.25,0.25],
                 *args, **kwargs):
        """
        add_plot(lon, lat, data, **kwargs)
        
        Inputs:
        ==============
        sub : subplot (this is returuned from add_plot())
        ax. : axis number to add colorbar to
        
        """
        
        self.vrange = vrange
        self.ncolors = ncolors
        self.cmap = cmap
        
        ### Set Longitude if none is given
        if lon is None:
            self.lon = np.arange(0.5,360,1)
        else:
            self.lon = lon
            
        ### Set latitude if none is given
        if lat is None:
            self.lat = np.arange(-89.5,90,1)
        else:
            self.lat = lat
            
        self.transform = ccrs.PlateCarree(central_longitude=0)
        self.bounds = np.linspace(self.vrange[0], self.vrange[1], self.ncolors)
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        # Define southern ocean region
        if self.region.upper()=='SOUTHERN-OCEAN':
            # Compute a circle in axes coordinates, which we can use as a boundary
            # for the map. We can pan/zoom as much as we like - the boundary will be
            # permanently circular.
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            
            # Set extent
            self.grid[ax].set_boundary(circle, transform=self.grid[ax].transAxes)

            # Limit the map to -60 degrees latitude and below.
            self.grid[ax].set_extent([-180, 180, -90, -35], ccrs.PlateCarree())
            
            
        ### land mask
        # Add Contintents
        if land is True:
            self.grid[ax].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                edgecolor='None', 
                                                facecolor=facecolor))
        
        ## add Coastline
        if coastline is True:
            self.grid[ax].coastlines(facecolor=facecolor, linewidth=linewidth_coast)
        
        sub = self.grid[ax].pcolormesh(self.lon, self.lat, data,
                            norm=self.norm,
                            transform=self.transform,
                            cmap = self.cmap,
                            vmin = self.vrange[0],
                            vmax = self.vrange[1], *args, **kwargs)
        return sub
    
    def add_colorbar(self, sub, ax=0, *args, **kwargs):
        """
        add_colorbar(sub, ax, **kwargs)
        
        Inputs:
        ==============
        sub : subplot (this is returuned from add_plot())
        ax. : axis number to add colorbar to
        
        """
        # Weird whitespace when you use 'extend'
        # The workaround is to make a colorbar
        # Help from : https://github.com/matplotlib/matplotlib/issues/9778

        #col = self.grid.cbar_axes[ax].colorbar(sub, *args, **kwargs)
        col = mpl.colorbar.ColorbarBase(self.grid.cbar_axes[ax], 
                                        orientation=self.cbar_orientation,
                                        cmap=self.cmap,
                                        norm=mpl.colors.Normalize(vmin=self.vrange[0], 
                                                                  vmax=self.vrange[1]),
                                        *args, **kwargs)

        return col
    
    ### Class methods
    def set_ticks(self, col, tmin, tmax, dt, *args, **kwargs):
        """
        set_ticks(tmin,tmax,dt, **kwargs)
        
        Inputs:
        ==============
        tmin : min tick value
        tmax : max tick value
        dt.  : delta tick value
        
        """
        col.cbar_axis.set_ticks(np.arange(tmin, tmax+dt, dt), *args, **kwargs)
        
    def set_title(self, title, ax, *args, **kwargs):
        """
        set_title(title, *args, **kwargs)
        
        Inputs:
        ==============
        title : title value
        
        """
        self.grid[ax].set_title(title, *args, **kwargs)
        
    def set_cbar_title(self, col, title, *args, **kwargs):
        """
        set_cbar_title(title, *args, **kwargs)
        
        Inputs:
        ==============
        title : colorbar title value
        
        """
        col.ax.set_title(title, *args, **kwargs)

    def set_cbar_ylabel(self, col, ylabel, *args, **kwargs):
        """
        set_cbar_ylabel(title, *args, **kwargs)
        
        Inputs:
        ==============
        title : colorbar title value
        
        """
        col.ax.set_ylabel(ylabel, *args, **kwargs)
        
    def set_cbar_xlabel(self, col, ylabel, *args, **kwargs):
        """
        set_cbar_xlabel(title, *args, **kwargs)
        
        Inputs:
        ==============
        title : colorbar title value
        
        """
        col.ax.set_xlabel(ylabel, *args, **kwargs)
        
    def set_cbar_xticklabels(self, col, labels, *args, **kwargs):
        """
        set_cbar_labels(labels, *args, **kwargs)
        
        Inputs:
        ==============
        labels : custom colorbar labels
        
        """
        col.ax.set_xticklabels(labels, *args, **kwargs)
        
    def set_cbar_yticklabels(self, col, labels, *args, **kwargs):
        """
        set_cbar_labels(labels, *args, **kwargs)
        
        Inputs:
        ==============
        labels : custom colorbar labels
        
        """
        col.ax.set_yticklabels(labels, *args, **kwargs)
