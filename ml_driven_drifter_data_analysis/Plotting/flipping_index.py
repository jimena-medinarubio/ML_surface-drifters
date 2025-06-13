
#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
from haversine import haversine, Unit
import sys
import seaborn as sns
# %%
PROJ_ROOT = DATA_DIR = '/Users/1614576/Desktop/ML-driven drifter data analysis/' 

interpolated_data_file= f'{PROJ_ROOT}/data/interim/interpolated_atm_ocean_datasets_depth.nc'
save_file=f'{PROJ_ROOT}/reports/figures/FigE2.svg'
# %%
# Load the interpolated data
dt_obs=xr.open_datatree(interpolated_data_file)

#create figure to plot mean flipping index over all drifters as a function of time
plt.figure(figsize=(12, 6))
flipping_list = []
for drifter in dt_obs.children:
    try:
        da = dt_obs[drifter].ds['flipping_index']
        # Convert time to days since release
        time_since_release = (da.time - da.time[0]) / np.timedelta64(1, 'D')
        da = da.assign_coords(days_since_release=("time", time_since_release.data))
        flipping_list.append(da)
    except KeyError:
        print(f"Skipping {drifter} (no flipping_index)")

# Step 2: Interpolate all to a common "days since release" grid
min_days = max([da.days_since_release.min().values for da in flipping_list])
max_days = min([da.days_since_release.max().values for da in flipping_list])
common_time= np.arange(min_days, max_days, 1/24)  # adjust resolution here (e.g. 0.1 = every ~2.4 hours)

flipping_swapped = [da.swap_dims({"time": "days_since_release"}) for da in flipping_list]

# Step 3: Interpolate each flipping_index to the common time grid
aligned = [da.interp(days_since_release=common_time) for da in flipping_swapped]

# Step 4: Combine into a single DataArray
flipping_all = xr.concat(aligned, dim='drifter')

# Step 5: Plot
plt.figure(figsize=(9, 5))

# Individual drifters
for i in range(flipping_all.drifter.size):
    plt.plot(common_time, flipping_all.isel(drifter=i), color='#EC7014', alpha=0.3)

# Mean
mean_flipping = flipping_all.mean(dim='drifter')
plt.plot(common_time, mean_flipping, color='black', linewidth=2, label='Average')

plt.xlabel('Time since release [days]', fontsize=14)
plt.ylabel('Flipping Index', fontsize=14)
plt.legend( fontsize=14)
plt.tight_layout()
plt.savefig(save_file, dpi=300)
plt.show()
# %%
