import numpy as np
import pandas as pd

#import h3
from h3 import h3

import pytide

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry.polygon import Polygon

# various plotting parameter

plt_params = {'extent':[-75,20,-70,70],
              'dticks':[20,20],
              'projection': ccrs.PlateCarree(0)}
#              'projection': ccrs.Robinson(central_longitude=-30)}
#              'projection': ccrs.Mollweide()}


# ------------------------------------------------------------------------------------

# harmonic analysis

def harmonic_analysis(df, ssh_key, min_count=100, constituents=None):

    if constituents is None:
        _cst = ["M2", "K1", "O1", "P1", "Q1", "S1"]
    else:
        _cst = constituents
                
    if df.empty or df.size<min_count:
        
        return pd.DataFrame([[0. for c in _cst]], columns=_cst)
        
    else:
        
        # preliminary info
        time = df['time'].to_numpy(dtype="datetime64[us]")
        wt = pytide.WaveTable(_cst)
        f, vu = wt.compute_nodal_modulations(time)

        # get harmonics
        w = wt.harmonic_analysis(df[ssh_key].to_numpy(), f, vu)
        
        # predicted tidal contribution
        #hp = wt.tide_from_tide_series(time, w)

        return pd.DataFrame([w], columns=_cst)
        

# ------------------------------------------------------------------------------------

# h3 code

def get_hex(row, resolution, *args, **kwargs):
    return h3.geo_to_h3(row["latitude"], row["longitude"], resolution)

def add_lonlat(df, reset_index=False):
    if reset_index:
        df = df.reset_index()
    df['lat'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[0])
    df['lon'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[1])
    return df
    
def id_to_bdy(hex_id):
    hex_boundary = h3.h3_to_geo_boundary(hex_id) # array of arrays of [lat, lng]                                                                                                                                                                                                                                                         
    hex_boundary = hex_boundary+[hex_boundary[0]]
    return [[h[1], h[0]] for h in hex_boundary]

def plot_h3_simple(df, metric_col, x='lon', y='lat', marker='o', alpha=1, 
                 figsize=(16,12), colormap='viridis'):
    df.plot.scatter(x=x, y=y, c=metric_col, title=metric_col
                    , edgecolors='none', colormap=colormap, 
                    marker=marker, alpha=alpha, figsize=figsize);
    plt.xticks([], []); plt.yticks([], [])

def plot_h3(df, metric_col, vmin=None, vmax=None, 
            x='lon', y='lat', marker='o', alpha=1, s=3**2,
            figsize=(10,10), 
            colorbar=True, colormap='plasma_r', colorbar_kwargs={},
            scatter=True,
            **kwargs):
    #
    _projection = plt_params['projection']
    _extent = plt_params['extent']
    _dticks = plt_params['dticks']
    #
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=_projection)
    ax.set_extent(_extent)
    #
    if isinstance(metric_col, str):
        _df = df
        _mcol = metric_col
    else:
        assert len(metric_col)==2
        _df = df[metric_col[0]].join([df['lon'], df['lat']])
        _mcol = metric_col[1]
    #
    if scatter:
        im = _df.plot.scatter(x=x, y=y, c=_mcol, s=s,
                        title=_mcol,
                        vmin=vmin, vmax=vmax,
                        ax = ax,
                        edgecolors='none', colormap=colormap, 
                        marker=marker, alpha=alpha, figsize=figsize,
                        transform=ccrs.PlateCarree(), **kwargs)
    else:
        if vmin is None:
            vmin = _df[_mcol].min()
        if vmax is None:
            vmax = _df[_mcol].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(colormap)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for index, row in _df.iterrows():
            pgon = Polygon(id_to_bdy(row['hex_id']))
            ax.add_geometries([pgon], crs=ccrs.PlateCarree(),
                              facecolor=m.to_rgba(row[_mcol]),
                              edgecolor=None)
        
    plt.xticks([], []); plt.yticks([], [])
    
    #if colorbar:
    #    cbar = fig.colorbar(im, **colorbar_kwargs)
    #else:
    #    cbar = None
    # grid lines:
    xticks = np.arange(_extent[0],
                       _extent[1]+_dticks[0],
                       _dticks[1]*np.sign(_extent[1]-_extent[0]))
    ax.set_xticks(xticks,crs=ccrs.PlateCarree())
    yticks = np.arange(_extent[2],
                       _extent[3]+_dticks[1],
                       _dticks[1]*np.sign(_extent[3]-_extent[2]))
    ax.set_yticks(yticks,crs=ccrs.PlateCarree())
    gl = ax.grid()
    ax.add_feature(cfeature.LAND)
    #
    #if title is not None:
    #    ax.set_title(title,fontdict={'fontsize':20, 'fontweight':'bold'})
    
# to download cartopy data, on datarmor frontend:
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#_extent = [-75,20,-70,70]
#_dticks = [20,20]
#_projection = ccrs.PlateCarree(0)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection=_projection)
#ax.set_extent(_extent)
#ax.add_feature(cfeature.LAND)


# ------------------------------------------------------------------------------------

# plot data as outputed from xhistogram

def plot_xhist(v, vmin=None, vmax=None, x='longitude_bin', y='latitude_bin', 
               figsize=(10,10), colorbar=True, colmap=None, colorbar_kwargs={}):
    _projection = plt_params['projection']
    _extent = plt_params['extent']
    _dticks = plt_params['dticks']
    #
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=_projection)
    ax.set_extent(_extent)
    im = v.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
                            x=x, y=y, cmap=colmap, add_colorbar=False)
    if colorbar:
        cbar = fig.colorbar(im, **colorbar_kwargs)
    else:
        cbar = None
    # grid lines:
    xticks = np.arange(_extent[0],
                       _extent[1]+_dticks[0],
                       _dticks[1]*np.sign(_extent[1]-_extent[0]))
    ax.set_xticks(xticks,crs=ccrs.PlateCarree())
    yticks = np.arange(_extent[2],
                       _extent[3]+_dticks[1],
                       _dticks[1]*np.sign(_extent[3]-_extent[2]))
    ax.set_yticks(yticks,crs=ccrs.PlateCarree())
    gl = ax.grid()
    ax.add_feature(cfeature.LAND)
    #
    #if title is not None:
    #    ax.set_title(title,fontdict={'fontsize':20, 'fontweight':'bold'})
