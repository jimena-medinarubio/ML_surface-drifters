#%%
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from pathlib import Path
from haversine import haversine, Unit
#%%
#specify directories
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCES_DIR = PROJ_ROOT / "references"


#define drifter file
drifter_file='waddendrifters2024.nc'
# define drifter ID file
drifter_id_file='drifters_info.csv'
# velocity data
velocity_file='external/velocity_land_mask.nc'

#open datasets
ds_og=xr.open_dataset(f'{RAW_DATA_DIR}/{drifter_file}')
df_drifters = pd.read_csv(f'{REFERENCES_DIR}/{drifter_id_file}',  delimiter=';', dtype={'ID': str})
drifters_id_dict = dict(zip(df_drifters['Drifter'], df_drifters['ID']))
#land mask
velocity_land_mask = xr.open_dataset(f'{DATA_DIR}/{velocity_file}')

#create xarray DataTree from the original dataset
ds={str(df_drifters['ID'].values[i]): ds_og.sel(trajectory=df_drifters['Drifter'].values[i]).swap_dims({"obs": "time"}) for i in range(len(ds_og.trajectory.values))}
dt_og = xr.DataTree.from_dict(ds)

# %%

def remove_after_land(dt, velocity_ds, buffer_time=500, initial_time=True):
    """
    Remove all time coordinates from all datasets after the first land encounter.

    dt: xarray.DataTree with drifter coordinates
    velocity_ds: xarray.Dataset with velocity data to create land mask
    buffer_time: int, time in minutes to allow before first land encounter to filter out intertidal zone
    initial_time: bool, if True, allow initial buffer time of 24h for analysis
    """
    #create land mask from velocity dataset where velocity is NaN
    land_mask = np.where(np.isnan(velocity_ds['uo'][0]), 0, 1)
    mask = RegularGridInterpolator((velocity_ds['latitude'].values, velocity_ds['longitude'].values), land_mask, method='nearest')
    
    first_land_times, last_measurement_times = {}, {}

    for key in dt.leaves:
        ds = key.ds
        lons, lats =ds['lon'].values, ds['lat'].values
        time_steps = np.sum(~np.isnan(ds['time'].values))

        for t in range(time_steps):
            if mask((lats[t], lons[t])) == 0:
                first_land= ds['time'][t].values
                first_land_times[key.name] = first_land
                if t>buffer_time:
                    break
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

        if initial_time==True:
            dictionary[key.name] = ds.isel(time=slice(60*24//5, index + 1)) #allow initial buffer time of 24h
        else:
            dictionary[key.name] = ds.isel(time=slice(0, index + 1))
    
    return xr.DataTree.from_dict(dictionary)

# %%
def filter_time(dt, threshold=2.5):
    """
    Filter out time steps where the time difference is larger than the threshold.

    dt: xarray.DataTree with drifter coordinates
    threshold: float, maximum allowed time difference in seconds between consecutive observations.
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

    dt: xarray.DataTree with drifter coordinates.
    threshold: float, maximum allowed instantaneous speed (in consistent units with distance/time).
    """

    datasets = {}
    for node in dt.leaves:  # Iterate over all leaf nodes
        ds = node.ds  # Extract the dataset from the node
        lon = ds['lon'].values
        lat = ds['lat'].values
        time = ds['time']

        # Compute time differences in seconds 
        time_diffs = time.diff('time').dt.total_seconds().values  # Ensure correct shape
        distances = [haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS) for lat1, lon1, lat2, lon2 in zip(lat[:-1], lon[:-1], lat[1:], lon[1:])]
        # Compute speeds (m/s)
        speeds = distances / time_diffs
        # Identify indices where the speed is less than or equal to the threshold
        valid_indices = np.where(speeds <= threshold)[0] + 1  # Correct the indexing
        # Apply filtering
        valid_indices = np.insert(valid_indices, 0, 0)  # Add the first index
        datasets[node.name] = ds.isel(time=valid_indices)
    
    return xr.DataTree.from_dict(datasets)

#%%

#apply funcitons to data
dt = remove_after_land(dt_og, velocity_land_mask, initial_time=False)
dt = filter_time(dt)
dt = eliminate_signal_errors(dt)
#%%
# save pre-processed DataTree to a NetCDF file
dt.to_netcdf(f'{PROJ_ROOT}/data/interim/preprocessed_drifter_data.nc')  
# %%
