#%%
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from haversine import haversine, Unit
from scipy.ndimage import gaussian_filter1d
#%%
import sys
sys.path.append("..")
from ml_driven_drifter_data_analysis.plots_old import plot_trajs
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCES_DIR = PROJ_ROOT / "references"
#%%

#define pre-processed file
drifter_file=f'{DATA_DIR}/interim/preprocessed_drifter_data.nc'
# define drifter ID file
drifter_id_file='drifters_info.csv'
#%%
#open datasets
dt_og=xr.open_datatree(f'{RAW_DATA_DIR}/{drifter_file}')
df_drifters = pd.read_csv(f'{REFERENCES_DIR}/{drifter_id_file}',  delimiter=';', dtype={'ID': str})

#%%
def drifter_velocity(data):
    """compute zonal and meridional velocities of drifter
    data : dictionary containing arrays time, lat and lon
    u, v : array, zonal (meridional) velocity
    """
    Re = 6371000 # radius earth (m)
    time = (data['time'] - data['time'][0]).astype('timedelta64[s]')
    latr = np.deg2rad(data['lat'])
    lonr = np.deg2rad(data['lon'])

    u = Re*np.cos(latr)*np.gradient(lonr,time)
    v = Re*np.gradient(latr,time)

    return u,v
#%%
def calculate_velocities(dt, method='cds'):
    """
    Calculate time differences, distances, and zonal/meridional displacements, then compute forward and 
    central velocities for each trajectory in the xarray dataset.
    
    ds (xarray.Dataset): Dataset with 'lon', 'lat', and 'time' variables along with 
                                       'trajectory' and 'obs' dimensions.
    method: 'cds' (central difference scheme), 'forward' (forward differences)
    
    Returns:
    xarray.Dataset: Dataset with added velocity variables.
    """

    d={}
    
    for node in dt.leaves:
        ds=node.ds
        
        # Compute time differences in seconds between observations
        time_diffs=ds['time'].differentiate('obs', datetime_unit='s')
        # Compute haversine distances
        dl = [haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS) 
              for lat1, lon1, lat2, lon2 in zip(ds['lat'].values[:-1], ds['lon'].values[:-1], ds['lat'].values[1:], ds['lon'].values[1:])]

        vel_x, vel_y= drifter_velocity(ds)
        speeds=dl/ time_diffs[1:].values
        speeds = np.concatenate((speeds, [speeds[-1]]))

    # Convert the DatasetView to a mutable dataset
        mutable_ds = ds.copy()

        # Apply selected method to compute velocities
        if method == 'cds':  # Central Difference Scheme
            # Compute central velocities using xarray shift method
            vel_x_central = (vel_x[1:] + vel_x[:-1] ) / 2
            vel_y_central = (vel_y[1:] + vel_y[:-1]) / 2
            speeds_central = (speeds[1:] + speeds[:-1]) / 2

            vel_x_central=np.insert(vel_x_central, [0, -1], [vel_x_central[0], vel_x_central[-1]])
            vel_y_central=np.insert(vel_y_central, [0], vel_y_central[0])
            speeds_central=np.insert(speeds_central, [0,], speeds_central[0])

            mutable_ds["vx"] =  xr.DataArray(vel_x_central, dims="time", coords={"time": mutable_ds["time"]})
            mutable_ds["vy"] = xr.DataArray(vel_y_central, dims="time", coords={"time": mutable_ds["time"]})
            mutable_ds["v"] = xr.DataArray(speeds_central, dims="time", coords={"time": mutable_ds["time"]})
            #plt.plot(mutable_ds['time'], mutable_ds['vx'])

        elif method == 'forward':  # Forward Difference Scheme
            mutable_ds["vx"] = xr.DataArray(vel_x, dims="time", coords={"time": mutable_ds["time"]})
            mutable_ds["vy"] = xr.DataArray(vel_y, dims="time", coords={"time": mutable_ds["time"]})
            mutable_ds["v"] = xr.DataArray(speeds, dims="time", coords={"time": mutable_ds["time"]})

        d[node.name] = mutable_ds

    return xr.DataTree.from_dict(d)
#%%
def calcualte_residual(dt, period='24.83h'):

    """
    Calculate the residual velocities for each drifter in the dataset.

    dt: xarray.DataTree
    period: str, rolling window period for calculating the residual velocities (default is M2 period).
        
   """

    d={}
    for node in dt.leaves:
        ds=node.ds
        mutable_ds = ds.copy()

        time_drifter = pd.DatetimeIndex(ds['time'].values)
        drifter_df = pd.DataFrame({var:ds[var].values for var in ds.variables}, index=time_drifter)
        dx=drifter_df['vx'].rolling(window=period, center=True).mean()
        dy=drifter_df['vy'].rolling(window=period, center=True).mean()

        mutable_ds['vy_residual']=xr.DataArray(dy, dims="time")
        mutable_ds['vx_residual']=xr.DataArray(dx, dims="time")

        d[node.name]=mutable_ds
    
    return xr.DataTree.from_dict(d) 
#%%
def select_frequency_interval(ds, freqs, atol=25, window_size=100):
    
    """
    calculate the last index in the time series where the time difference is stable within a specified frequency range.

    ds: xarray.Dataset with 'time' variable
    freqs: float, the frequency to check stability against (in seconds)
    window_size: int, the size of the rolling window to calculate the moving average of time differences (default is 100)
    """
    # Calculate the time differences in seconds
    time_diff = ds['time'].differentiate('obs', datetime_unit='s')
    # Initialize an array to store the moving average of time differences
    rolling_mean = np.convolve(time_diff, np.ones(window_size)/window_size, mode='valid')
    stable_intervals = np.isclose(rolling_mean, freqs, atol=atol)
    ## Get the last index in the rolling mean where the interval is stable
    last_true_in_rolling = np.where(stable_intervals)[0][-1]        
    # Adjust the index to match the original time_diff length
    last_true_index = last_true_in_rolling + window_size - 1  # Adjust for the window size offset
    return last_true_index

#%%
def flipping_index_continuous(dt, deltat, sampling_frequencies=[300, 1800]):

    """
    Calculate a continuous flipping index for each drifter and over the trajectory dimensions.
    The window size is adapted based on the time difference between measurements.

    dt: xarray.DataTree with drifter data
    deltat: float, time difference in hours between measurements
    sampling_frequencies: list of frequencies in seconds to calculate flipping index for
    """
    d={}

    for node in dt.leaves:
        ds = node.ds.copy()
        # Create a new DataArray to store flipping_index_continuous
        flipping_index_array = xr.DataArray(np.full(ds['time'].shape, np.nan), dims='time', coords={'time': ds['time']})

        for i, dt in enumerate(sampling_frequencies):
            start_idx = select_frequency_interval(ds, sampling_frequencies[i - 1]) * i
            end_idx = select_frequency_interval(ds, dt)
            
            number_timesteps = int(deltat * 3600 / dt)  # Convert delta t from hours to timesteps

            # Calculate flips where orientation changes (0 ↔ 1)
            flips =np.abs(np.diff( ds['orientation'].isel(time=slice(start_idx, end_idx)).values))
            flips=xr.DataArray(flips, dims='time')
   
            # Apply rolling sum to count flips over a window of size equal to the number of timesteps
            flipping_index_discrete = flips.rolling(time=number_timesteps, center=True, min_periods=1).sum()

            # Compute sigma 
            sigma = len(ds['obs'].isel(time=slice(start_idx, end_idx))) / (2*np.sum(flips))  # Prevent division by zero

            flipping_index_continuous=gaussian_filter1d(flipping_index_discrete.values, sigma=sigma.values)
            # Assign values back to the array
            flipping_index_array[start_idx:end_idx] = np.pad(flipping_index_continuous, (0, end_idx - start_idx - len(flipping_index_continuous)), mode='edge')

        ds['flipping_index'] = flipping_index_array
        d[node.name]=ds
    
    max_fi_dict = {drifter: float(max(data['flipping_index'])) for drifter, data in d.items()}

    for drifter in d:
        d[drifter]['flipping_index_scaled'] =xr.DataArray(np.array(d[drifter]['flipping_index'])/max(max_fi_dict.values()), dims='time', coords={'time': d[drifter]['time']})
    
    return xr.DataTree.from_dict(d)
#%%
#apply functions
dt=calculate_velocities(dt_og)
dt=calcualte_residual(dt)
dt=flipping_index_continuous(dt, 3, sampling_frequencies=[300, 1800])

# %%
#save datatree to netcdf file
dt.to_netcdf(f'{PROJ_ROOT}/data/interim/processed_drifter_data.nc')  

# %%
