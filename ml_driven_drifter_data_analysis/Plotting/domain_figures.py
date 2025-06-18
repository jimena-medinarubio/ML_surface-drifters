#%%

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import cmocean
import seaborn as sns

import sys
sys.path.append('..')
from config import FIGURES_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

#%%
def plot_trajs_bathymetry_int(
    dt, 
    german_bight_bathymetry,  
    name_fig='plot', 
    output_path=None, 
    palette=None, 
    output_format='svg'):


    fig = plt.figure(figsize=(15, 18))
    ax_inset = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Main map region (German Bight)
    ax_inset.set_extent([3, 9, 53, 56], crs=ccrs.PlateCarree())

    gb_bathy = -german_bight_bathymetry['deptho'].values
    gb_bathy[np.isnan(gb_bathy)] = 0
    mesh = ax_inset.pcolormesh(
        german_bight_bathymetry['longitude'], 
        german_bight_bathymetry['latitude'], 
        gb_bathy, 
        cmap=cmocean.cm.ice,
        transform=ccrs.PlateCarree(),
        zorder=1)
    ax_inset.add_feature(cfeature.LAND, color='lightgray', zorder=2)
    ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

    if palette is None:
       palette = palette = ["#d699ff", "#9a0809", "#8848cd", "#e9b856",
           "#f0e442", "#936a54", "#009e3d", "#b3d500", '#4b5d9a', "#ea780a", "#eb9cc0", "#101088"]

    for idx, node in enumerate(dt.leaves):
        drifter = node.ds
        ax_inset.plot(
            drifter['lon'], drifter['lat'], 
            color=palette[idx % len(palette)],
            linewidth=1.5, 
            zorder=4
        )
        ax_inset.scatter(
            drifter['lon'].values[0], drifter['lat'].values[0], 
            marker='*', s=150, 
            color= palette[idx % len(palette)],
            edgecolor='black', zorder=5,
        )
        ax_inset.scatter(
            drifter['lon'].values[-1], drifter['lat'].values[-1], 
            marker='^', s=150, 
            color= palette[idx % len(palette)],
            edgecolor='black', zorder=5,
        )
    ax_inset.scatter(
            [], [], 
            marker='*', s=150, color='black',
            label='Start'
        )
    ax_inset.scatter([], [],
            marker='^', s=150, 
            label='End', color='black'
        )


    ax_inset.legend(
        loc='upper right', framealpha=1, edgecolor='black', fontsize=14
    )


    # Colorbar for bathymetry plot
    cbar = fig.colorbar(mesh, ax=ax_inset, orientation='vertical', pad=0.02, shrink=0.35)
    cbar.set_label('Depth [m]', fontsize=10)
    cbar.ax.tick_params(labelsize=12)

    # Add inset map with Orthographic projection (on the bottom-right corner)
    ax_robinson = fig.add_axes([0.64, 0.28, 0.12, 0.28], projection=ccrs.Orthographic(central_longitude=6, central_latitude=54.5))

    # Plot the world map with grey land and blue ocean
    ax_robinson.add_feature(cfeature.LAND, color='lightgrey', zorder=1)
    ax_robinson.add_feature(cfeature.OCEAN, color='lightblue', zorder=2)
    ax_robinson.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

    # Draw a red rectangle to indicate the region of interest (German Bight)
    gb_lon = [8.3, 8.35, 8.35, 8.3, 8.3]
    gb_lat = [53.4, 53.4, 53.45, 53.45, 53.4]
    # Correcting the transform for the Orthographic plot (centered at 6, 54.5)
    #ax_inset.plot(gb_lon, gb_lat, color='red', linewidth=2, zorder=4)

    ax_rec = fig.add_axes([0.695, 0.34, 0.012, 0.16],projection=ccrs.PlateCarree())

    ax_rec.plot(gb_lon, gb_lat,
                 color='red',
                 linewidth=2,
                 transform=ccrs.PlateCarree(),  # critical!
                 zorder=10)
    ax_rec.patch.set_alpha(0)
    ax_rec.axis('off')

    # Set ticks for the Orthographic projection
    ax_inset.set_xticks(np.arange(4, 9, 1))
    ax_inset.set_yticks(np.arange(53.5, 56, 0.5))
    ax_inset.set_xticklabels([f'{i}°E' for i in np.arange(4, 9, 1)])  # Longitude ticks
    ax_inset.set_yticklabels([f'{i}°N' for i in np.arange(53.5, 56, 0.5)])  # Latitude ticks
    ax_inset.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    # Save or show
    if output_path is None:
        output_path = FIGURES_DIR 
    
    plt.savefig(f'{output_path}/{name_fig}.{output_format}', dpi=300, format=f'{output_format}', bbox_inches='tight')
    plt.show()

# %%

bathymetry_file=xr.open_dataset(f'{EXTERNAL_DATA_DIR}/bathymetry.nc')
dt_og=xr.open_datatree(f'{INTERIM_DATA_DIR}/preprocessed_drifter_data_noinitialtime.nc')
#%%
plot_trajs_bathymetry_int(dt_og, bathymetry_file, 'Fig1', output_path=None, palette=None, output_format='svg')
# %%
