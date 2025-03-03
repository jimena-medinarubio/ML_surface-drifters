#%%
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from pathlib import Path
from ml_driven_drifter_data_analysis.plots import plot_trajs
#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

ds_og=xr.open_dataset(f'{RAW_DATA_DIR}/waddendrifters2024.nc')
df_drifters = pd.read_csv(f'{PROJ_ROOT}/references/drifters_id.csv',  dtype={'ID': str})
drifters_id_dict = dict(zip(df_drifters['Drifter'], df_drifters['ID']))

ds={str(df_drifters['ID'].values[i]): ds_og.sel(trajectory=df_drifters['Drifter'].values[i]).swap_dims({"obs": "time"}) for i in range(len(ds_og.trajectory.values))}
dt = xr.DataTree.from_dict(ds)

# %%

def remove_after_land(dt, velocity_ds):
    """
    Remove all time coordinates from all datasets after the first land encounter.
    """
    land_mask = np.where(np.isnan(velocity_ds['uo'][0]), 0, 1)
    mask = RegularGridInterpolator((velocity_ds['latitude'].values, velocity_ds['longitude'].values), land_mask, method='nearest')
    
    first_land_times, last_measurement_times = {}, {}
    for key in dt.leaves:
        ds = key.ds
        lons, lats =ds['lon'].values, ds['lat'].values
        time_steps = np.sum(~np.isnan(ds['time'].values))
        
        for t in range(time_steps):
            if mask((lats[t], lons[t])) == 0:
                first_land= ds['time'].values[t]
                first_land_times[key.name] = first_land
            
        last_measurement_times[key.name]=ds['time'].values[time_steps-1]

    last_index = min(min(first_land_times.values()), min(last_measurement_times.values()))
    print(f'Time first drifter beaches: {min(first_land_times.values())}')
    print(f'Time first drifter stops measuring: {min(last_measurement_times.values())}')
    
    dictionary = {}
    for key in dt.leaves:
        ds = key.ds
        time_index = pd.to_datetime(ds['time'].values)
        # Filter out 'naT' values
        time_array = time_index[~time_index.isna()].values.astype('datetime64')
        differences = np.abs(time_array - last_index)
        index = np.argmin(differences)
        dictionary[key.name] = ds.isel(time=slice(0, index + 1))
    
    return xr.DataTree.from_dict(dictionary)

# %%
def filter_time(dt, threshold=2.5):
    """
    Filter out time steps where the time difference is larger than the threshold.
    """
    dtc = dt.copy()  # Copy the tree to avoid modifying the original
    for node in dt.leaves:  # Iterate over all leaf nodes
        ds = node.ds  # Extract the dataset from the node
        time_diff = ds["time"].diff("time").dt.total_seconds()
        indices = np.where(time_diff > threshold)[0] + 1  # Shift by 1 to match subsequent time steps
        # Ensure the first time step is included
        indices = np.insert(indices, 0, 0)  # Add the first index since diff excludes it
        
        # Assign filtered dataset back to the node
        dtc[node.name] = ds.isel(time=indices)
    
    return dtc

# %%
def eliminate_signal_errors(dt, threshold=3):
    """
    Remove signal errors by filtering out observations where the speed 
    between consecutive points exceeds a given threshold.

    Parameters:
    dt (xarray.DataTree): A datatree with datasets containing dimensions ('obs', 'trajectory') and variables 'lon', 'lat', 'time'.
    threshold (float): Maximum allowed speed (in consistent units with distance/time).

    Returns:
    xarray.DataTree: A new datatree with erroneous observations removed.
    """

    def haversine(lon1, lat1, lon2, lat2):
        """Compute the great-circle distance between two points using the Haversine formula."""
        R = 6371e3  # Earth radius in meters
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c  # Distance in meters

    datasets = {}
    for node in dt.leaves:  # Iterate over all leaf nodes
        ds = node.ds  # Extract the dataset from the node
        lon = ds['lon'].values
        lat = ds['lat'].values
        time = ds['time']

        # Compute time differences in seconds 
        time_diffs = time.diff('time').dt.total_seconds().values  # Ensure correct shape
        distances = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        # Compute speeds (m/s)
        speeds = distances / time_diffs
        # Identify indices where the speed is less than or equal to the threshold
        valid_indices = np.where(speeds <= threshold)[0] + 1  # Correct the indexing
        # Apply filtering
        valid_indices = np.insert(valid_indices, 0, 0)  # Add the first index
        datasets[node.name] = ds.isel(time=valid_indices)
    
    return xr.DataTree.from_dict(datasets)

#%%
#land mask
velocity_land_mask = xr.open_dataset(f'{PROJ_ROOT}/data/external/velocity_land_mask.nc')

#apply funcitons to data
dt = remove_after_land(dt, velocity_land_mask)
dt = filter_time(dt)
dt = eliminate_signal_errors(dt)

# Save or further process the DataTree (dt) as needed
dt.to_netcdf(f'{PROJ_ROOT}/data/interim/processed_data.nc')  # Example placeholder for saving dt
# %%
