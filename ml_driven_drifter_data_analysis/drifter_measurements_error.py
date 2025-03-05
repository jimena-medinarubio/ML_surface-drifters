#%%

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

dt=xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
# %%

def find_static_coords(dt, time_buffer=50, name_fig=None, output_path=None):

    def degrees_to_meters(deg_lat, deg_lon, reference_lat, Re=6371e3):
        """Convert degrees to meters using haversine approximation."""
        lat_meters = (np.pi / 180) * Re * deg_lat
        lon_meters = (np.pi / 180) * Re * np.cos(np.radians(reference_lat)) * deg_lon
        return lat_meters, lon_meters

    bright_palette = sns.color_palette("bright", 10)
    extra_colors = [(204/255, 153/255, 255/255), (0, 153/255, 153/255)]   # Bright Teal
    # Extend the original palette by appending the new colors
    palette = bright_palette + extra_colors

    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Latitude deviation [m]', size=14)
    ax1.set_ylabel('KDE',size=14)
    ax1.tick_params(axis='both', labelsize=14)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Longitude deviation [m]', size=14)
    ax2.set_ylabel('KDE',size=14)
    ax2.tick_params(axis='both', labelsize=14)

    for i, node in enumerate(dt.leaves):
        ds=node.ds
        ds=ds.where(ds['v'][:time_buffer]<0.01, drop=True)
        deviations_lat, deviations_lon= degrees_to_meters(ds['lat'], ds['lon'], ds['lat'].mean())

        latitude_std=deviations_lat.std().values
        longitude_std=deviations_lon.std().values
        print(latitude_std)

        sns.kdeplot(deviations_lat - np.mean(deviations_lat), color=palette[i], ax=ax1, fill=True, alpha=0.1, common_norm=False, label=rf'$\sigma={np.round(latitude_std, 2)}$')
        sns.kdeplot(deviations_lon - np.mean(deviations_lon), color=palette[i], ax=ax2, fill=True, alpha=0.1, common_norm=False, label=rf'$\sigma={np.round(longitude_std, 2)}$')

    ax1.legend(fontsize=14)
    ax2.legend(fontsize=14)

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path_lon= FIGURES_DIR / f'{name_fig}_lon.png'
        output_path_lat= FIGURES_DIR / f'{name_fig}_lat.png'
    fig1.savefig(output_path_lat, dpi=300)
    fig2.savefig(output_path_lon, dpi=300)

    plt.show()

# %%
find_static_coords(dt, name_fig='error_drifter_measurements')
# %%
