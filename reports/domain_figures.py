#%%

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import cmocean

#%%

def plot_bathymetry(bathymetry_field, name_fig, output_path=None):
    df = {'deptho': bathymetry_field['deptho'].values}
    # Replace NaN values with 0
    df['deptho'][np.isnan(df['deptho'])] = 0

    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolormesh(bathymetry_field['longitude'], bathymetry_field['latitude'], df['deptho'], cmap=cmocean.cm.deep , 
                   vmax=80, zorder=1, transform=ccrs.PlateCarree() )
    cbar = plt.colorbar(label='Depth [m]', ax=ax, shrink=0.4)
    cbar.ax.tick_params(labelsize=12)  # Adjust tick label size
    cbar.set_label('Depth [m]', fontsize=14)  # Adjust label font size

    # Add land and ocean features
    land_feature = cfeature.LAND  # Reference to LAND feature
    ax.add_feature(land_feature, edgecolor='w', facecolor='lightgrey', zorder=2)  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=4)


    # Set the tick locations explicitly for longitude and latitude axes
    ax.set_xticks(np.arange(4, 9, 1))
    ax.set_yticks(np.arange(53.5, 56, 0.5))
    ax.set_xticklabels([f'{i}°E' for i in np.arange(4, 9, 1)])  # Longitude ticks
    ax.set_yticklabels([f'{i}°N' for i in np.arange(53.5, 56, 0.5)])  # Latitude ticks
    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    plt.xlim(3, 9)
    plt.ylim(53, 56)
    # Show the plot
    plt.tight_layout()

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
#%%
def plot_europe_with_region(name_fig, output_path=None):
    # Create a map of Europe
    fig, ax = plt.subplots(figsize=(5, 6), subplot_kw={'projection': ccrs.Robinson()})
    ax.set_extent([-10, 30, 35, 70], crs=ccrs.PlateCarree())  # Extent for Europe

    # Add land and coastlines
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, zorder=2)

    # Draw the delimited region
    region_x = [3, 9, 9, 3, 3]  # Longitude coordinates of the square
    region_y = [53, 53, 56, 56, 53]  # Latitude coordinates of the square
    ax.plot(region_x, region_y, color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=3)
   # ax.text(6, 54.5, 'Study Area', color='red', transform=ccrs.PlateCarree(),
           # ha='center', fontsize=10, zorder=4)

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5)

    # Adjust gridline labels
    gl.xlabel_style = {'size': 12}  # Set font size for longitude labels
    gl.ylabel_style = {'size': 12}  # Set font size for latitude labels

    # Optionally adjust gridline appearance
    gl.xlines = False  # Remove vertical gridlines
    gl.ylines = False  # Remove horizontal gridlines

    # You can also choose whether to show the gridline labels at the top or right
    gl.top_labels = False
    gl.right_labels = False

    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes
    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.svg'
    plt.savefig(output_path, dpi=300)

    plt.show()

# %%

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

bathymetry_file=xr.open_dataset(f'{DATA_DIR}/external/bathymetry.nc')
plot_bathymetry(bathymetry_file, 'bathymetry')
#%%
plot_europe_with_region('europe')
# %%
